# Learned Index Structures

This is an implentation of using machine learning to index databases, as decribed in this research [paper](https://arxiv.org/abs/1712.01208). 

Databases are indexed depending on the type of query, point or range. Traditionally, hash maps are used for point queries, while B trees are for range queries. Both structures can be considered "models", as they map keys to indexes (locations in memory) that contain corresponding values. In this repository, we will explore nueral networks and other ML models to approximate index locations.
