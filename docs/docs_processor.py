# docs_processor.py
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocsProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = []
        self.embeddings = None
        self.debug = True  # Enable debugging

    def safe_read_json(self, file_path):
        """Safely read JSON file with different encoding attempts"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return json.load(f)
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error with {encoding} encoding: {str(e)}")
                continue
        
        raise ValueError(f"Could not read file {file_path} with any of the attempted encodings")

    def process_docs(self):
        try:
            dataset = self.safe_read_json('data/dataset_api.json')
            enrichment = self.safe_read_json('data/discovery_enrichment_api.json')
            
            self.docs = []
            
            def flatten_json(obj, prefix='', source=''):
                if isinstance(obj, dict):
                    # Add the entire object as context if it has certain key indicators
                    if any(key in obj for key in ['endpoint', 'description', 'parameters', 'example']):
                        self.docs.append(f"{source} - {prefix}: {json.dumps(obj, indent=2)}")
                    
                    for key, value in obj.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, (dict, list)):
                            flatten_json(value, new_prefix, source)
                        else:
                            self.docs.append(f"{source} - {new_prefix}: {str(value)}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_prefix = f"{prefix}[{i}]"
                        if isinstance(item, (dict, list)):
                            flatten_json(item, new_prefix, source)
                        else:
                            self.docs.append(f"{source} - {new_prefix}: {str(item)}")

            flatten_json(dataset, source='dataset')
            flatten_json(enrichment, source='enrichment')
            
            if self.debug:
                print(f"Processed {len(self.docs)} documentation chunks")
                print("Sample chunks:")
                for i in range(min(3, len(self.docs))):
                    print(f"Chunk {i}: {self.docs[i][:200]}...")

            if self.docs:
                self.embeddings = self.model.encode(self.docs)
            else:
                raise ValueError("No documentation content was processed")

        except Exception as e:
            raise Exception(f"Error processing documentation: {str(e)}")

    def find_relevant_context(self, query, top_k=8):  # Increased top_k to get more context
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = [self.docs[i] for i in top_indices]
        
        # Prioritize chunks with endpoint information
        endpoint_chunks = [chunk for chunk in relevant_chunks if 'endpoint' in chunk.lower()]
        parameter_chunks = [chunk for chunk in relevant_chunks if any(param in chunk.lower() for param in ['parameter', 'filter', 'type', 'value'])]
        
        context = "\n".join(endpoint_chunks + parameter_chunks)
        return context