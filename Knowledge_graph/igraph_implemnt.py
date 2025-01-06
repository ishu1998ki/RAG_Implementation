import json
import igraph as ig
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from inserter import insert_custom_kg_safely
from ER_extraction import entities_out
from umap import UMAP

data = entities_out()
print(data)

## Knowledge Graph Processing ##

def load_knowledge_graph(json_data: Dict) -> Tuple[List[str], List[Tuple[int, int, dict]]]:
    """
    Load knowledge graph data from JSON and prepare it for igraph conversion.

    Args:
        json_data: Dictionary containing entities and relationships

    Returns:
        Tuple containing:
        - List of unique entity names
        - List of edges with attributes
    """
    # Extract entities and create a mapping of names to indices
    entities = [entity['entity_name'].strip() for entity in json_data['entities']]
    entity_to_idx = {name: idx for idx, name in enumerate(entities)}

    # Process relationships into edge list with attributes
    edges = []
    for rel in json_data['relationships']:
        src = rel['src_id'].strip()
        tgt = rel['tgt_id'].strip()

        # Only add edge if both nodes exist
        if src in entity_to_idx and tgt in entity_to_idx:
            edge_attrs = {
                'description': rel['description'],
                'weight': float(rel['weight']) if rel['weight'] else 1.0
            }
            edges.append((entity_to_idx[src], entity_to_idx[tgt], edge_attrs))

    return entities, edges


def create_igraph(vertices: List[str], edges: List[Tuple[int, int, dict]]) -> ig.Graph:
    """
    Create an igraph Graph object from vertices and edges.

    Args:
        vertices: List of vertex names
        edges: List of tuples (source_idx, target_idx, attributes)

    Returns:
        igraph.Graph object
    """
    # Create directed graph
    g = ig.Graph(directed=True)

    # Add vertices
    g.add_vertices(len(vertices))
    g.vs['name'] = vertices

    # Add edges with attributes
    if edges:  # Check if there are any edges
        edge_tuples = [(e[0], e[1]) for e in edges]
        g.add_edges(edge_tuples)

        # Add edge attributes
        for idx, (_, _, attrs) in enumerate(edges):
            for key, value in attrs.items():
                if g.es.attribute_names().count(key) == 0:
                    g.es[key] = [None] * len(g.es)
                g.es[idx][key] = value

    return g

def personalized_pagerank_search(g: ig.Graph, query_vertices: List[str], damping: float = 0.85) -> Dict[str, float]:
    """
    Perform personalized PageRank search on the graph.

    Args:
        g: igraph Graph object
        query_vertices: List of vertex names to use as personalization vector
        damping: Damping factor for PageRank (default: 0.85)

    Returns:
        Dictionary mapping vertex names to their PageRank scores
    """
    # Create personalization vector by finding vertex indices
    reset_vertices = []
    for vertex in query_vertices:
        try:
            vertex_idx = g.vs.find(name=vertex).index
            reset_vertices.append(vertex_idx)
        except ValueError:
            continue

    # If no valid vertices found, use uniform distribution
    if not reset_vertices:
        return {v['name']: 1.0 / len(g.vs) for v in g.vs}

    # Calculate personalized PageRank
    weights = g.es.get_attribute_values('weight') if 'weight' in g.es.attributes() else None

    pagerank_scores = g.personalized_pagerank(
        weights=weights,
        damping=damping,
        reset_vertices=reset_vertices
    )

    # Create results dictionary
    results = {v['name']: score for v, score in zip(g.vs, pagerank_scores)}
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


def process_knowledge_graph(json_data: Dict, query_entities: List[str]) -> Dict[str, float]:
    """
    Process knowledge graph and perform personalized PageRank search.

    Args:
        json_data: Knowledge graph data in JSON format
        query_entities: List of entity names to search for

    Returns:
        Dictionary of entity names and their PageRank scores
    """
    # Convert JSON to igraph structure
    vertices, edges = load_knowledge_graph(json_data)
    graph = create_igraph(vertices, edges)

    # Perform personalized PageRank search
    results = personalized_pagerank_search(graph, query_entities)

    return results

## Improve Knowledge Graph Processing with similarity search ##

def load_knowledge_graph_with_embeddings(json_data: Dict) -> Tuple[List[str], List[Tuple[int, int, dict]], np.ndarray]:
    """
    Load knowledge graph data with entity embeddings for similarity calculations.
    """
    entities, edges = load_knowledge_graph(json_data)

    # Generate embeddings for entities
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(entities)

    return entities, edges, embeddings


def similarity_weighted_pagerank(g: ig.Graph,
                                 query_vertices: List[str],
                                 embeddings: np.ndarray,
                                 damping: float = 0.85,
                                 similarity_weight: float = 0.5) -> Dict[str, float]:
    """
    Combine PersonalizedPageRank with cosine similarity for ranking.
    """
    # Get embeddings for query vertices
    query_indices = []
    for vertex in query_vertices:
        try:
            idx = g.vs.find(name=vertex).index
            query_indices.append(idx)
        except ValueError:
            continue

    if not query_indices:
        return {v['name']: 1.0 / len(g.vs) for v in g.vs}

    # Calculate cosine similarities
    query_embedding = np.mean(embeddings[query_indices], axis=0).reshape(1, -1)
    similarities = cosine_similarity(embeddings, query_embedding).flatten()

    # Calculate PageRank scores
    weights = g.es.get_attribute_values('weight') if 'weight' in g.es.attributes() else None
    pagerank_scores = g.personalized_pagerank(
        weights=weights,
        damping=damping,
        reset_vertices=query_indices
    )

    # Combine scores
    combined_scores = {}
    for v, pr_score, sim_score in zip(g.vs, pagerank_scores, similarities):
        combined_scores[v['name']] = (1 - similarity_weight) * pr_score + similarity_weight * sim_score

    return dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))


def process_knowledge_graph_with_similarity(json_data: Dict,
                                            query_entities: List[str],
                                            similarity_weight: float = 0.5) -> Dict[str, float]:
    """
    Process knowledge graph with combined PageRank and similarity scoring.
    """
    vertices, edges, embeddings = load_knowledge_graph_with_embeddings(json_data)
    graph = create_igraph(vertices, edges)
    results = similarity_weighted_pagerank(
        graph,
        query_entities,
        embeddings,
        similarity_weight=similarity_weight
    )
    return results

## Improve Knowladge graph search with chain-of-thought ##

class ChainOfThoughtGraphSearch:
    def __init__(self, json_data: Dict, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.vertices, self.edges, self.embeddings = load_knowledge_graph_with_embeddings(json_data)
        self.graph = create_igraph(self.vertices, self.edges)
        self.reasoning_history: List[Dict] = []

    def iterative_search(self, query: str, max_steps: int = 3) -> Dict[str, float]:
        current_entities = set(self._initial_entity_extraction(query))
        all_relevant_entities: Set[str] = set()

        for step in range(max_steps):
            # Get current results
            step_results = similarity_weighted_pagerank(
                self.graph,
                list(current_entities),
                self.embeddings
            )

            # Update reasoning history
            self.reasoning_history.append({
                'step': step,
                'query_entities': current_entities,
                'top_results': dict(list(step_results.items())[:5])
            })

            # Expand search with new entities
            new_entities = self._extract_related_entities(step_results, threshold=0.3)
            all_relevant_entities.update(current_entities)
            current_entities = new_entities - all_relevant_entities

            if not current_entities:
                break

        return self._aggregate_results()

    def _initial_entity_extraction(self, query: str) -> List[str]:
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return [self.vertices[i] for i in np.argsort(similarities)[-3:]]

    def _extract_related_entities(self, results: Dict[str, float], threshold: float) -> Set[str]:
        return {entity for entity, score in results.items() if score > threshold}

    def _aggregate_results(self) -> Dict[str, float]:
        all_scores = {}
        decay_factor = 0.8

        for step, history in enumerate(self.reasoning_history):
            step_weight = decay_factor ** step
            for entity, score in history['top_results'].items():
                all_scores[entity] = all_scores.get(entity, 0) + score * step_weight

        return dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

    def get_reasoning_chain(self) -> List[Dict]:
        return self.reasoning_history


## Visualization ##

def create_plotly_graph_visualization(
        g: ig.Graph,
        title: str,
        pagerank_scores: Optional[Dict[str, float]] = None,
        highlight_vertices: Optional[List[str]] = None,
        edge_threshold: float = 0.0,
        reasoning_step: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Create an interactive plotly visualization of the graph with embedding-based layout.

    Args:
        g: igraph Graph object
        title: Title for the visualization
        pagerank_scores: Optional dictionary of vertex names to PageRank scores
        highlight_vertices: Optional list of vertex names to highlight
        edge_threshold: Minimum PageRank score for vertices to include
        reasoning_step: Optional step number in the reasoning chain
        embeddings: Optional numpy array of entity embeddings for layout

    Returns:
        plotly Figure object
    """
    # Use UMAP or PCA layout if embeddings are provided, otherwise use Kamada-Kawai
    if embeddings is not None:
        from umap import UMAP
        layout = UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        layout = g.layout_kamada_kawai()

    # Calculate color scheme based on scores
    if pagerank_scores:
        scores = np.array([pagerank_scores.get(v['name'], 0) for v in g.vs])
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        colorscale = px.colors.sequential.Viridis
        vertex_colors = [px.colors.sample_colorscale(colorscale, score)[0]
                         for score in normalized_scores]
    else:
        vertex_colors = ['lightblue'] * len(g.vs)

    # Adjust vertex sizes based on scores and highlighting
    vertex_sizes = [30] * len(g.vs)
    if pagerank_scores:
        max_score = max(pagerank_scores.values())
        for i, vertex in enumerate(g.vs):
            score = pagerank_scores.get(vertex['name'], 0)
            vertex_sizes[i] = 20 + (score / max_score) * 50

            if highlight_vertices and vertex['name'] in highlight_vertices:
                vertex_colors[i] = 'red'
                vertex_sizes[i] *= 1.5

    # Create edge traces with improved styling
    edge_traces = []
    for edge in g.es:
        source = g.vs[edge.source]
        target = g.vs[edge.target]

        if pagerank_scores:
            source_score = pagerank_scores.get(source['name'], 0)
            target_score = pagerank_scores.get(target['name'], 0)
            if source_score < edge_threshold or target_score < edge_threshold:
                continue

        x0, y0 = layout[edge.source]
        x1, y1 = layout[edge.target]

        # Calculate edge width based on weight
        weight = edge['weight'] if 'weight' in g.es.attributes() else 1.0
        edge_width = 0.5 + weight * 2

        # Correctly access edge description attribute
        description = edge['description'] if 'description' in g.es.attributes() else ''

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge_width, color='#888'),
            hoverinfo='text',
            text=description
        )
        edge_traces.append(edge_trace)

    # Create figure
    fig = go.Figure()

    # Add edges
    for trace in edge_traces:
        fig.add_trace(trace)

    # Add nodes with improved styling
    node_trace = go.Scatter(
        x=[layout[i][0] for i in range(len(g.vs))],
        y=[layout[i][1] for i in range(len(g.vs))],
        mode='markers+text',
        marker=dict(
            size=vertex_sizes,
            color=vertex_colors,
            line=dict(width=2, color='black'),
            symbol='circle'
        ),
        text=[f"{v['name']}<br>Score: {pagerank_scores.get(v['name'], 0):.4f}"
              if pagerank_scores else v['name'] for v in g.vs],
        textposition="top center",
        hoverinfo='text'
    )
    fig.add_trace(node_trace)

    # Update layout with reasoning step information
    title_text = f"{title}<br>Step {reasoning_step}" if reasoning_step is not None else title
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            y=0.95
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig


def create_reasoning_chain_visualization(
        search_instance: 'ChainOfThoughtGraphSearch',
        score_threshold: float = 0.01
) -> List[go.Figure]:
    """
    Create visualizations for each step in the reasoning chain.

    Args:
        search_instance: ChainOfThoughtGraphSearch instance
        score_threshold: Threshold for including vertices

    Returns:
        List of plotly figures, one for each reasoning step
    """
    figures = []

    for step, history in enumerate(search_instance.reasoning_history):
        step_fig = create_plotly_graph_visualization(
            search_instance.graph,
            f"Knowledge Graph Exploration - Step {step + 1}",
            history['top_results'],
            list(history['query_entities']),
            score_threshold,
            step + 1,
            search_instance.embeddings
        )
        figures.append(step_fig)

    return figures


def display_graph_visualizations(
        search_instance: 'ChainOfThoughtGraphSearch',
        score_threshold: float = 0.01
) -> Tuple[go.Figure, List[go.Figure]]:
    """
    Process knowledge graph and create full visualization plus reasoning chain.

    Args:
        search_instance: ChainOfThoughtGraphSearch instance
        score_threshold: Threshold for including vertices

    Returns:
        Tuple of (final_graph_figure, list_of_reasoning_step_figures)
    """
    # Create final aggregate visualization
    final_results = search_instance._aggregate_results()
    final_graph = create_plotly_graph_visualization(
        search_instance.graph,
        "Final Knowledge Graph Results",
        final_results,
        list(search_instance.reasoning_history[-1]['query_entities']),
        score_threshold,
        embeddings=search_instance.embeddings
    )

    # Create reasoning chain visualizations
    reasoning_steps = create_reasoning_chain_visualization(
        search_instance,
        score_threshold
    )

    return final_graph, reasoning_steps

query_entities = ["Romeo", "Juliet"]

results = process_knowledge_graph(data, query_entities)

print("\nTop 10 most relevant entities:[using first method]")
for entity, score in list(results.items())[:10]:
    print(f"{entity}: {score:.4f}")


results = process_knowledge_graph_with_similarity(data, query_entities)

results_semantic = process_knowledge_graph_with_similarity(data, query_entities, similarity_weight=0.7)

print("\nTop most relevant entities:[using second improved method]")
for entity, score in list(results.items())[:5]:
    print(f"{entity}: {score:.3f}")


# Create search instance
search = ChainOfThoughtGraphSearch(data)

# Perform search
results = search.iterative_search("Who are the key characters involved in the conflict between Montagues and Capulets?")

# Create visualizations
final_graph, reasoning_steps = display_graph_visualizations(search)

# Display figures (if using in a notebook)
# final_graph.show()
# for step_fig in reasoning_steps:
#     step_fig.show()


## Chunk retriever ##

class KnowledgeGraphChunkRetriever:
    def __init__(self, json_data: Dict):
        """
        Initialize the retriever with knowledge graph data.

        Args:
            json_data: Dictionary containing entities, relationships, and chunks
        """
        self.entities = [entity['entity_name'].strip() for entity in json_data['entities']]
        self.chunks = json_data['chunks']
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create chunk embeddings
        self.chunk_embeddings = self.model.encode([chunk['content'] for chunk in self.chunks])

        # Create entity to chunk mapping
        self.entity_chunks = self._create_entity_chunk_mapping()

    def _create_entity_chunk_mapping(self) -> Dict[str, Set[int]]:
        """Create a mapping of entities to chunk indices where they appear."""
        entity_chunks = {entity: set() for entity in self.entities}

        for chunk_idx, chunk in enumerate(self.chunks):
            chunk_content = chunk['content'].lower()
            for entity in self.entities:
                if entity.lower() in chunk_content:
                    entity_chunks[entity].add(chunk_idx)

        return entity_chunks

    def get_relevant_chunks(self, ranked_entities: Dict[str, float],
                            max_chunks: int = 5,
                            similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant chunks based on ranked entities.

        Args:
            ranked_entities: Dictionary of entity names and their importance scores
            max_chunks: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of relevant chunks with their scores
        """
        # Collect chunk indices from top entities
        relevant_chunk_indices = set()
        for entity in ranked_entities:
            relevant_chunk_indices.update(self.entity_chunks.get(entity, set()))

        if not relevant_chunk_indices:
            return []

        # Calculate chunk scores
        chunk_scores = []
        for chunk_idx in relevant_chunk_indices:
            score = self._calculate_chunk_score(
                chunk_idx,
                ranked_entities,
                self.chunk_embeddings[chunk_idx]
            )
            if score >= similarity_threshold:
                chunk_scores.append({
                    'chunk': self.chunks[chunk_idx],
                    'score': score,
                    'relevant_entities': [
                        entity for entity in ranked_entities
                        if chunk_idx in self.entity_chunks[entity]
                    ]
                })

        # Sort and return top chunks
        chunk_scores.sort(key=lambda x: x['score'], reverse=True)
        return chunk_scores[:max_chunks]

    def _calculate_chunk_score(self,
                               chunk_idx: int,
                               ranked_entities: Dict[str, float],
                               chunk_embedding: np.ndarray) -> float:
        """Calculate relevance score for a chunk based on entities and content."""
        # Entity presence score
        entity_score = sum(
            ranked_entities[entity]
            for entity in ranked_entities
            if chunk_idx in self.entity_chunks[entity]
        )

        # Normalize entity score
        if entity_score > 0:
            entity_score = entity_score / max(ranked_entities.values())

        return entity_score


def process_query_with_chunks(json_data: Dict,
                              query_entities: List[str],
                              graph_search_results: Dict[str, float]) -> List[Dict]:
    """
    Process a query and return relevant chunks based on graph search results.

    Args:
        json_data: Knowledge graph data
        query_entities: List of query entity names
        graph_search_results: Results from graph search algorithm

    Returns:
        List of relevant chunks with scores and metadata
    """
    retriever = KnowledgeGraphChunkRetriever(json_data)
    return retriever.get_relevant_chunks(graph_search_results)

# First perform graph search
search = ChainOfThoughtGraphSearch(data)
graph_results = search.iterative_search("Who are the key characters in the conflict?")

# Then retrieve relevant chunks
chunk_results = process_query_with_chunks(data, search.reasoning_history[-1]['query_entities'], graph_results)

print(chunk_results)



