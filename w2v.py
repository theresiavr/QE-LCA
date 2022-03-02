import pandas as pd

def find_top_n_terms(query, model, n, expansion_rep="w2v", pairwise=True):

    #split query
    list_query = query.split(" ")
    #check if query term is in vocab
    vocab_check = [term for term in list_query if term in model.wv.vocab]
    length = len(vocab_check)
    if length == 0:
        return []
    else:
        term_with_score = pd.DataFrame()
        for term in vocab_check:
            if expansion_rep == "w2v":
                result = model.wv.most_similar([term], topn=100)
            elif expansion_rep == "d2v":
                result = model.docvecs.most_similar([model.wv[term]], topn=100)
            else:
                raise Exception
#             print(result)
            term_with_score = term_with_score.append(result)
        term_with_score.columns=["term", "score"]
#        print(term_with_score)
        final_result = term_with_score.groupby("term").mean()
        sorted_final_result = final_result.sort_values(by="score",ascending=False)
        sorted_final_result = sorted_final_result.iloc[:n]
        return sorted_final_result["score"].to_dict()

