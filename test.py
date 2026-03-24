#!/usr/bin/env python3

import git_filter_repo as fr

# 构造参数列表（模拟命令行参数）
args = fr.FilteringOptions.parse_args(
    [
        "--force",
        "--path",
        "python-rag/data/vector_store/faiss_index/index.faiss",
        "--path",
        "python-rag/data/vector_store/faiss_index/index.pkl",
        "--path",
        "python-rag/data/corpus/combined_corpus.json",
        "--path",
        "python-rag/data/corpus/statpearls/statpearls_articles.json",
        "--invert-paths",
    ]
)

# 创建过滤器并运行
repo_filter = fr.RepoFilter(args)
repo_filter.run()
