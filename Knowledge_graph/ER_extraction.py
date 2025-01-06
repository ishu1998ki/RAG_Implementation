import os
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# OpenAI api define
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o",base_url=os.getenv("base_url"))

# Groq api define
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Define the directory containing the text file
file_path = os.path.join("./books", "romeo_and_juliet_copy.txt")
# print(file_path)

loader = TextLoader(file_path,encoding='UTF-8')
documents = loader.load()

rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)

rec_char_docs = rec_char_docs[:10]

# for index, char in enumerate(rec_char_docs):
#     if index == 0:  # Check if it's the first item
#         print(char)  # Print the first item
#         break


# Pydatntic class definition
# class for named_entity definition
class EntityExtraction(BaseModel):
    entities: str = Field(description="all the entities in a provided content")


# class for named_entity relationship structure
class NER_structure(BaseModel):
    Subject: str = Field(description="The entity being described.")
    Predicate: str = Field(description="The property or relationship associated with the subject.")
    Object: str = Field(description=" The value or entity to which the predicate applies.")


# class for get named_entity relationships
class EntityAndRelationship_Extraction(BaseModel):
    relationship: list[NER_structure] = Field(description="these are the relationship between extracted named entities")


def entities_out():
    # extract entities
    entity_structured_output_model = llm.with_structured_output(schema=EntityExtraction)

    # extract named_entities along with the relationships
    relationship_structured_output_model = llm.with_structured_output(schema=EntityAndRelationship_Extraction)

    entities_output = []
    relationship_output = []

    for chunk in rec_char_docs:
        entities_output.append(entity_structured_output_model.invoke(
            [
                {"role": "system",
                 "content": """
                 Named entities are special names of people, places, or things that stand out in a sentence. You need to extract named entities from the given content. Get idea from below provided examples.
    
                  Example 1: 
                  'In Silicon Valley, tech giants like Apple and Google create new technologies. Apple Park, their headquarters in Cupertino, is super cool.'
    
                  Named entities: Silicon Valley (a place), Apple (a company), Google (a company), Apple Park (a place), Cupertino (a place)
    
                  Example 2: 
                  'In New York City, the Statue of Liberty is a symbol of freedom. It was a gift from France and sits on Liberty Island, near the Hudson River'
    
                  Named entities: New York City (a place), Statue of Liberty (a famous statue), France (a country), Liberty Island (a place), Hudson River (a river)
    
                  Example 3:
                  'In Paris, the Eiffel Tower remains an iconic symbol of French culture and engineering brilliance. Lots of people visit the Louvre Museum to see amazing art like the Mona Lisa'
    
                  Named entities: Paris (a place), Eiffel Tower (a famous building), Louvre Museum (a place), Mona Lisa (a famous painting)
    
                    """},
                {"role": "user",
                 "content": chunk.page_content}
            ]))
    # print(entities_output)

    for i, chunk in enumerate(rec_char_docs):
        relationship_output.append(relationship_structured_output_model.invoke(
            [{"role": "system",
              "content": """Extract the relationship between identified named entities from the given content
                  Example 1: 
                  'In Silicon Valley, tech giants like Apple and Google create new technologies. Apple Park, their headquarters in Cupertino, is super cool'
    
                  Relationship:
                  Apple Park is the headquarters of Apple, located in Cupertino, within Silicon Valley.
    
    
                  Example 2: 
                  'In New York City, the Statue of Liberty is a symbol of freedom. It was a gift from France and sits on Liberty Island, near the Hudson River'
    
                  Relationships: 
                  Statue of Liberty is located in New York City and situated on Liberty Island.
                  Statue of Liberty was a gift from France to the United States.
                  Liberty Island, where the Statue of Liberty stands, overlooks the Hudson River. 
    
    
                  Example 3:
                  'In Paris, the Eiffel Tower remains an iconic symbol of French culture and engineering brilliance. Lots of people visit the Louvre Museum to see amazing art like the Mona Lisa'
    
                  Relationships: 
                  Eiffel Tower is an iconic landmark located in Paris, symbolizing French culture and engineering brilliance. 
                  Louvre Museum is situated in Paris and houses the famous Mona Lisa, attracting art enthusiasts.
              """},

             {"role": "user",
              "content": chunk.page_content + entities_output[i].entities}]))

    # print(relationship_output)


    formated_entities=[]

    for entity in entities_output:
        for _ in entity.entities.split(","):
            ent = {"entity_name":_,
                   "entity_type":"",
                   "description":"",
                   "source_id":""
                   }
            formated_entities.append(ent)


    formated_relationship=[]

    for relationship in relationship_output:
        for _ in relationship.relationship:
            relation = {"src_id":_.Subject,
                        "tgt_id":_.Object,
                        "description":_.Predicate,
                        "keywords":"",
                        "weight":1.0,
                        "source_id":""
                        }
            formated_relationship.append(relation)


    formated_chunks = []
    for chunk in rec_char_docs:
        _ch = {
            "content":chunk.page_content,
            "source_id":""
        }
        formated_chunks.append(_ch)


    custom_kg = {"entities":formated_entities,
                 "relationships":formated_relationship,
                 "chunks":formated_chunks}
    print(custom_kg)
    return custom_kg
