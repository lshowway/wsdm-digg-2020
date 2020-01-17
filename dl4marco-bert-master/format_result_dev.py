def read_reranking_index():
    r2 = "D:/backup/wsdm_cup/reranking.index"
    rerank_index = []
    with open(r2, encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split('\t')
            tmp = list(map(int, tmp))
            rerank_index.append(tmp)
    return rerank_index

def read_labels():
    r1 = "D:/backup/wsdm_cup/ms_citation_data/qrel.dev.tsv"
    id_label = {}
    with open(r1, encoding='utf-8') as f:
        for line in f:
            id, _, label, _= line.strip().split('\t')
            id_label[id] = label
    return id_label

def compute_rerank_results(papers, index, label):
    for i in index:
        if i > len(papers) - 1:
            continue
        if papers[i] == label:
            return 1 / (i+1)
    return 0.0


def compute_BM25_results(papers, label):
    for i, paper in enumerate(papers):
        if paper == label:
            return 1 / (i+1)
    return 0.0


def read_BM25_results(rerank_index, id_label):
    r1 = "D:/backup/wsdm_cup/bm25-key-sentence-k1-default-bdefault-top3-res.csv"
    w1 = "D:/backup/wsdm_cup/ms_citation_result/dev-bm25-neg.prediction"
    fw = open(w1, 'w', encoding='utf-8')
    with open(r1, encoding='utf-8') as f:
        lines = f.readlines()
        end = 0
        total_score_1, total_score_2 = 0.0, 0.0
        count = 0
        K = 9
        for i, (line, index) in enumerate(zip(lines, rerank_index)):
            tmp = line.strip().split(',')
            tmp = tmp[:K]
            description_id = tmp[0]
            papers = tmp[1:K+1]
            label = id_label[description_id]
            # fw.write(description_id + ',' + label + '\t' + papers[index[0]] + ',' + papers[index[1]] + ',' + papers[index[2]] + '\n')
            fw.write(description_id + ',' + label + ',' + ','.join(papers) + '\n')
            print(i)
            end = i
            count += 1

            r1 = compute_BM25_results(papers, label)
            r2 = compute_rerank_results(papers, index, label)
            total_score_1 += r1
            total_score_2 += r2
        print("BM25 MAP@3: ", total_score_1 / count)
        print("rerank MAP@3: ", total_score_2 / count)

        # for line in lines[end+1:]:
        #     fw.write(line)

def read_csv(filename):
    cnt = 0
    with open(filename, encoding='utf-8') as f:
       while 1:
           line = f.readline()
           cnt += 1
           if not line:
               break
    print(cnt)


if __name__ == "__main__":
    # rerank_index = read_reranking_index()
    # id_label = read_labels()
    # read_BM25_results(rerank_index, id_label)
    filename = 'D:/backup/wsdm_cup/triples.train.small.tsv'
    read_csv(filename)