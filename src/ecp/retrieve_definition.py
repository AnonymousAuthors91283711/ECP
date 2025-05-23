# search_theorem_fuzzy_namespaced.py
from sentence_transformers import SentenceTransformer
import pickle
from rapidfuzz import fuzz
import faiss
import os
PREFERRED_NAMESPACES = [
    "Nat", "Int", "Rat", "Real", "Complex", "ENat", "NNReal", "EReal", "Monoid",
    "CommMonoid", "Group", "CommGroup", "Ring", "CommRing", "Field", "Algebra",
    "Module", "Set", "Finset", "Fintype", "Multiset", "List", "Fin", "BigOperators",
    "Filter", "Polynomial", "SimpleGraph.Walk", "Equiv", "Embedding", "Injective",
    "Surjective", "Bijective", "Order", "Topology"
]

def load_partitioned_dict(coarse=True):
    path = "data/retrieval_data/preferred_partitioned.pkl" if coarse else "data/retrieval_data/partitioned_theorems.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)
THEOREM_DICT = load_partitioned_dict()
def split_full_name(full_name: str):
    parts = full_name.split(".")
    return ".".join(parts[:-1]), parts[-1]

def search_theorem_fuzzy(query: str, top_n=5, same_namespace_n=3, coarse=True):
    partitioned_dict = load_partitioned_dict(coarse=coarse)

    query = query.strip().lower()
    query_ns, query_def = split_full_name(query)

    main_results = []
    same_ns_results = []

    seen_def_names = set()

    for namespace, subdict in partitioned_dict.items():
        for full_name, definition in subdict.items():
            ns, defn = split_full_name(full_name)
            ns_lc, defn_lc = ns.lower(), defn.lower()

            score_name = fuzz.ratio(defn_lc, query_def)
            score_ns = fuzz.ratio(ns_lc, query_ns)

            # Tier logic
            if defn_lc == query_def:
                if ns_lc.startswith(query_ns):
                    tier = 1
                elif ns_lc.endswith(query_ns):
                    tier = 2
                else:
                    tier = 3
            elif ns_lc == query_ns:
                tier = 4
            else:
                tier = 5

            tie_breaker = (
                -score_ns,
                -score_name,
                len(ns),
                len(defn),
            )

            key = definition["definition_name"]

            # Collect same-namespace results separately
            if ns_lc == query_ns:
                same_ns_results.append(((score_name, tie_breaker), definition))
            else:
                main_results.append(((tier, *tie_breaker), definition))

    # Sort and select
    main_results.sort(key=lambda x: x[0])
    same_ns_results.sort(key=lambda x: (-x[0][0], x[0][1]))  # sort by best name match in same namespace

    top_main = []
    for _, definition in main_results:
        name = definition["definition_name"]
        if name not in seen_def_names:
            top_main.append(definition)
            seen_def_names.add(name)
        if len(top_main) >= top_n:
            break

    top_same_ns = []
    for _, definition in same_ns_results:
        name = definition["definition_name"]
        if name not in seen_def_names:
            top_same_ns.append(definition)
            seen_def_names.add(name)
        if len(top_same_ns) >= same_namespace_n:
            break

    return top_main + top_same_ns

# CACHE_FILE = "data/retrieval_data/lean_definitions_embedding.pkl"
# # Function to compute or load cached embeddings
# def compute_embeddings(theorem_dict=THEOREM_DICT, model_name="all-MiniLM-L6-v2", cache_file=CACHE_FILE):
#     """Compute and cache theorem embeddings using Sentence Transformers."""
    
#     if os.path.exists(cache_file):  # Check if cache exists
#         print("Loading cached embeddings...")
#         with open(cache_file, 'rb') as f:
#             data = pickle.load(f)
#             return data['model'], data['theorem_names'], data['theorem_descs'], data['desc_embeddings'], data['faiss_index']
    
#     print("Computing embeddings (this will take time)...")
#     model = SentenceTransformer(model_name)

#     # Convert descriptions to a list
#     theorem_names = list(theorem_dict.keys())
#     theorem_descs = list(theorem_dict.values())

#     # Compute embeddings
#     desc_embeddings = model.encode(theorem_descs, normalize_embeddings=True)

#     # FAISS index for fast retrieval
#     dim = desc_embeddings.shape[1]  
#     faiss_index = faiss.IndexFlatIP(dim)  
#     faiss_index.add(desc_embeddings)

#     # Cache results
#     with open(cache_file, 'wb') as f:
#         pickle.dump({
#             'model': model,
#             'theorem_names': theorem_names,
#             'theorem_descs': theorem_descs,
#             'desc_embeddings': desc_embeddings,
#             'faiss_index': faiss_index
#         }, f)
    
#     return model, theorem_names, theorem_descs, desc_embeddings, faiss_index

# model, theorem_names, theorem_descs, desc_embeddings, faiss_index = compute_embeddings(THEOREM_DICT)

# # Embedding Search Function
# def search_theorem_embedding(query, model=model, theorem_names=theorem_names, faiss_index=faiss_index, top_n=5, theorem_dict=THEOREM_DICT):
#     """Search theorems based on semantic similarity using embeddings."""
#     query_embedding = model.encode([query], normalize_embeddings=True)
#     _, indices = faiss_index.search(query_embedding, top_n * 2)  # Search extra to filter duplicates

#     seen_def_names = set()
#     results = []

#     for i in indices[0]:
#         name = theorem_names[i]
#         definition = theorem_dict[name]
#         def_name = definition["definition_name"]
#         if def_name not in seen_def_names:
#             seen_def_names.add(def_name)
#             results.append(definition)
#         if len(results) >= top_n:
#             break

#     return results

# # Example usage
# if __name__ == "__main__":
#     query = "Set.card"
#     results = search_theorem_fuzzy(query, top_n=5, coarse=True)

#     for r in results:
#         print(f"{r['definition_name']}")
#         print(f"Type: {r['type_signature']}")
#         if r.get("description"):
#             print(f"Description: {r['description']}")
#         print()
#     results = search_theorem_embedding(query, top_n=5)
    
#     for r in results:
#         print(r)
