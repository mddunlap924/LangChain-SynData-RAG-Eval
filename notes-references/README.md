<h1 align="center">
  <!-- <a href="https://github.com/mddunlap924/VHSpy">
    <img src="https://raw.githubusercontent.com/mddunlap924/PyVHS/main/doc/imgs/pyvhs.png" width="512" height="256" alt="pyvhs">
  </a> -->
  Information Retrieval and RAG
</h1>

<p align="center">Richer details and references for NLP-based Information Retrieval and RAG systems.
</p> 

<p align="center">
<a href="#prompt-engineering">Prompt Engineering</a> &nbsp;&bull;&nbsp;
<a href="#background">Background</a> &nbsp;&bull;&nbsp;
<a href="#benefits">Benefits</a> &nbsp;&bull;&nbsp;
<a href="#prompt-templates">Prompt Templates</a> &nbsp;&bull;&nbsp;
<a href="#example-notebooks">Example Notebooks</a> &nbsp;&bull;&nbsp;
<a href="#issues">Issues</a> &nbsp;&bull;&nbsp;
<a href="#todos">TODOs</a>
</p>

# Prompt Engineering
- [Prompt Engineering Guide](https://promptingguide.ai/)
    - [Generating Synthetic Dataset for RAG](https://www.promptingguide.ai/applications/synthetic_rag)
    ```
    Task: Identify a counter-argument for the given argument.
    Argument #1: {insert passage X1 here}
    A concise counter-argument query related to the argument #1: {insert manually prepared query Y1 here}
    Argument #2: {insert passage X2 here}
    A concise counter-argument query related to the argument #2: {insert manually prepared query Y2 here}
    <- paste your examples here ->
    Argument N: Even if a fine is made proportional to income, you will not get the equality of impact you desire. This is because the impact is not proportional simply to income, but must take into account a number of other factors. For example, someone supporting a family will face a greater impact than someone who is not, because they have a smaller disposable income. Further, a fine based on income ignores overall wealth (i.e. how much money someone actually has: someone might have a lot of assets but not have a high income). The proposition does not cater for these inequalities, which may well have a much greater skewing effect, and therefore the argument is being applied inconsistently.
    A concise counter-argument query related to the argument #N:
    ```
- [Improving Search Ranking with Few-Shot Prompting of LLMs](https://blog.vespa.ai/improving-text-ranking-with-few-shot-prompting/)
```
These are examples of queries with sample relevant documents for
each query. The query must be specific and detailed.

Example 1:
document: $document_example_1
query: $query_example_1

Example 2:
document: #document_example_2
query: $query_example_2

Example 3:
document: $document_example_3
query: $query_example

Example 4:
document: $input_document
query:
```

- [LangChain MultiQueryRetiever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)

```The MultiQueryRetriever automates the process of prompt tuning by using an LLM to generate multiple queries from different perspectives for a given user input query. For each query, it retrieves a set of relevant documents and takes the unique union across all queries to get a larger set of potentially relevant documents. By generating multiple perspectives on the same question, the MultiQueryRetriever might be able to overcome some of the limitations of the distance-based retrieval and get a richer set of results.```

# Consistency Filtering / Query Consistency
- [InPars-v2: Large Language Models as Efficient
Dataset Generators for Information Retrieval](https://arxiv.org/pdf/2301.01820.pdf): refer to Section 2 and the second paragraph.
    
    ```Once the synthetic queries are generated, we apply a filtering step to select query-document pairs that are more likely to be relevant to each other. In InPars-v1, this filtering step consisted of selecting the top 10k query-document pairs with the highest log probabilities of generating a query given the 3-shot examples and the document as input. In InPars-v2, we use monoT5-3B [4] already fine tuned on MS MARCO for one epoch1 to estimate a relevancy score for each of the 100k query-document pairs. Then, we keep only the top 10k pairs with the highest scores as our positive query-document pairs for training.```
- Vespa AI
    - [Improving Zero-Shot Ranking with Vespa Hybrid Search](https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa/#next-blog-post-in-this-series)
    - [Improving Zero-Shot Ranking with Vespa Hybrid Search - part two](https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa-part-two/)
    - [Improving Search Ranking with Few-Shot Prompting of LLMs](https://blog.vespa.ai/improving-text-ranking-with-few-shot-prompting/)
        
        ```LLMs will hallucinate and sometimes produce fake queries that are too generic or irrelevant. To overcome this, researchers 1 2 3 4 use a ranking model (RM) to test the query quality. One can, for example, rank documents from the corpus for each generated synthetic query using the RM. The synthetic query is only retained for model training if the source document ranks highly. This query consistency check grounds the LLM output and improves the training data. Once the synthetic queries and positive (relevant) document pairs are filtered, one can sample negative (potentially irrelevant) examples and train a ranking model adapted to the domain, the characteristics of the document corpus, and the few-show query examples. We use a robust zero-shot hybrid ranking model for query consistency checking. The generated query is retained for training only if the source document is ranked #1 by the zero-shot model.```

- [Promptagator: Few-shot Dense Retrieval From 8 Examples](https://arxiv.org/pdf/2209.11755v1.pdf) Refer to Section 3.2 describing a retriever model to keep relevant passages...

  ```The filtering step improves the quality of generated queries by ensuring the round-trip consistency (Alberti et al., 2019): a query should be answered by the passage from which the query was generated. In our retrieval case, the query should retrieve its source passage. Consistency filtering (Alberti et al., 2019; Lewis et al., 2021) has been shown crucial for synthetic question generation on QA tasks.```

# Blogs
- [Zero and Few Shot Text Retrieval and Ranking Using Large Language Models](https://blog.reachsumit.com/posts/2023/03/llm-for-text-ranking/)
- [The ABCs of semantic search in OpenSearch: Architectures, benchmarks, and combination strategies](https://opensearch.org/blog/semantic-science-benchmarks/)

# Tutorials
- [PineCone: Unsupervised Training of Retrievers Using GenQ](https://www.pinecone.io/learn/series/nlp/genq/): This approach to building bi-encoder retrievers uses the latest text generation techniques to synthetically generate training data. In short, all we need are passages of text. The generation model then augments these passages with synthetic queries, giving us the exact format we need to train an effective bi-encoder model.

# GitHub
- [Inquisitive Parrots for Search: InPARs](https://github.com/zetaalphavector/InPars): A toolkit for end-to-end synthetic data generation using LLMs for IR
- [ir_datasets](https://github.com/allenai/ir_datasets): s a python package that provides a common interface to many IR ad-hoc ranking benchmarks, training datasets, etc.
- [BEIR Benchmarking IR](https://github.com/beir-cellar/beir): a heterogeneous benchmark containing diverse IR tasks. It also provides a common and easy framework for evaluation of your NLP-based retrieval models within the benchmark.
- [GitHub: Semantic Retrieval Models](https://github.com/caiyinqiong/Semantic-Retrieval-Models): A curated list of awesome papers for Semantic Retrieval, including some early methods and recent neural models for information retrieval tasks (e.g., ad-hoc retrieval, open-domain QA, community-based QA, and automatic conversation).

# Papers
- [Can Generative LLMs Create Query Variants for Test Collections?](https://www.microsoft.com/en-us/research/uploads/prod/2023/05/srp0313-alaofi.pdf)
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/pdf/2104.08663.pdf)
- [Questions Are All You Need to Train a Dense Passage Retriever ](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00564/116466/Questions-Are-All-You-Need-to-Train-a-Dense)