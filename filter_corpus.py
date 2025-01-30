import pandas as pd

def filter_corpus(corpus_path):
    relevant_vocabularies = {"SNOMED", "ICD10CM"}
    
    # Load only necessary columns to save memory
    usecols = ["concept_id", "concept_name", "vocabulary_id"]
    
    corpus_df = pd.read_csv(corpus_path, dtype=str, usecols=usecols, low_memory=False)
    
    # Filter for relevant vocabularies
    filtered_corpus = corpus_df[corpus_df["vocabulary_id"].isin(relevant_vocabularies)]
    
    print(f"Filtered corpus size: {filtered_corpus.shape[0]} rows (from {corpus_df.shape[0]})")
    
    return filtered_corpus

# Example usage
corpus_path = "test/vocabulary_reduced_SNOMED_LOINC_ICD9CM_ICD10CM_OMOP.csv"
filtered_corpus = filter_corpus(corpus_path)
filtered_corpus.to_csv('test/corpus_REDUCED.csv', index=False)