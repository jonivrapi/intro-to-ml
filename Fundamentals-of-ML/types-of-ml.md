---
order: 99
label: Types of ML
---

# Types of Machine Learning

Machine Learning (ML) is a vast field with various techniques and approaches. One of the fundamental ways to categorize these approaches is by how an algorithm learns to become more accurate in its predictions. There are four primary types of ML: Supervised Learning, Semisupervised Learning, Unsupervised Learning, and Reinforcement Learning. Each type has its unique approach and application areas.

## Supervised Learning

Supervised learning is the most common type of machine learning. In this approach, the algorithm learns from labeled training data, helping the model make predictions or decisions based on new, unseen data.

### How it Works:

-   **Training Data**: Involves input-output pairs. The input data is the data point, and the output is the label.
-   **Learning Process**: The algorithm learns by comparing its output with the actual answer and then adjusting itself to improve accuracy.
-   **Examples**:
    -   **Regression**: Predicting continuous values (e.g., house prices).
    -   **Classification**: Categorizing data into predefined classes (e.g., spam detection in emails).
-   **Algorithms**
    -   k-Nearest Neighbors
    -   Linear Regression
    -   Logistic Regression
    -   Support Vector Machines (SVMs)
    -   Decision Trees and Random Forests
    -   Neural Networks

## Semisupervised Learning

Semisupervised learning is supervised learning but over partially labelled data. Labeling data, in general, is very time-consuming, and you will frequently find yourself with many unlabeled datapoints and few labeled ones. Some algorithms are capable of dealing with this.

Of these algorithms, most are combinations of unsupervised and supervised algorithms. Google Photos is a good example of this. You upload an entire album of photos, and it is able to recognize that Person A is in photos 1, 2, and 3, while Person B is in photos 3, 4, and 5. Once you tell it that Person A is Bob and Person B is Mary, it is able to automatically label all photos of both people.

-   **Algorithms**
    -   Deep Belief Networks (DBNs)

## Unsupervised Learning

Unsupervised learning deals with data that has no labels. The goal here is to explore the data and find some form of structure or pattern.

### How it Works:

-   **Training Data**: Only the input data is available; there are no corresponding output labels.
-   **Learning Process**: The algorithm tries to organize the data in some way to describe its structure. This can mean grouping the data into clusters or finding different ways of looking at complex data.
-   **Examples**:
    -   **Clustering**: Grouping customers based on purchasing behavior.
    -   **Dimensionality Reduction**: Reducing the number of variables in a dataset while retaining its essential features.
-   **Algorithms**
    -   Clustering
        -   K-Means
        -   DBSCAN
        -   Hierarchical Cluster Analysis (HCA)
    -   Anomaly Detection and Novelty Detection
        -   One-class SVM
        -   Isolation Forest
    -   Dimensionality Reduction and Visualization
        -   Principal Component Analysis (PCA)
        -   Kernel PCA
        -   Locally Linear Embedding (LLE)
        -   t-Distributed Stochastic Neighbor Embedding (t-SNE)
    -   Association Rule Learning
        -   Apriori
        -   Eclat

## Reinforcement Learning

Reinforcement Learning is a bit different; it is about action and reaction. It’s learning what to do and how to map situations to actions to maximize a reward signal.

### How it Works:

-   **Training Data**: The algorithm interacts with a dynamic environment in which it must perform a certain goal (e.g., driving a car).
-   **Learning Process**: The model learns to achieve a goal in an uncertain, potentially complex environment. It uses feedback from its own actions and experiences.
-   **Examples**:
    -   **Game Playing**: AlphaGo, the program that plays Go.
    -   **Robot Navigation**: Robots learning to navigate through various terrains.

## Conclusion

Understanding these three types of machine learning is fundamental to diving deeper into the field. Each type has its unique challenges and requires different approaches and techniques. Other sections of these docs will have concrete examples of these, as well as more in-depth information.
