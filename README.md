# Trimester 1 - APEX Project: Customer Segmentation using Clustering

## Project Overview

This project aims to perform **customer segmentation** using various clustering techniques like **K-Means**, **Hierarchical Clustering**, and **HDBSCAN**. The project involves multiple steps, including **data extraction**, **preprocessing**, **clustering**, and **visualization**.

## Project Structure

The project consists of several Jupyter notebooks, each performing different tasks in the workflow. Below is the list of key notebooks and their purpose:

1. **00_load_customer_vectors.ipynb**: Loads customer vector embeddings, which are used to represent customers based on their behavior.
2. **00_make_train_test_split.ipynb**: Splits the dataset into training and testing sets to evaluate the clustering model performance.
3. **01_k_means_clustering.ipynb**: Applies the K-Means clustering algorithm to group customers into clusters based on their behavior.
4. **02_hier_segments.ipynb**: Uses **Hierarchical Clustering** to generate customer segments based on different distance metrics.
5. **03_umap_hdbscan.ipynb**: Combines **UMAP** for dimensionality reduction and **HDBSCAN** for clustering to identify customer segments.
6. **04_compare_clusterings.ipynb**: Compares the performance of K-Means, Hierarchical Clustering, and HDBSCAN using different evaluation metrics.
7. **04_visualization_and_storytelling.ipynb**: Visualizes the clustering results and performs storytelling using the clusters for analysis.
8. **cluster_busters_end_to_end_customer_segmentation.ipynb**: An end-to-end notebook for customer segmentation, combining all techniques in a single workflow.
9. **data_extraction_instacart.ipynb**: Extracts and processes the Instacart dataset to prepare it for clustering.
10. **Preprocessing_Feature_Engineering_the_cluster_busters.ipynb**: Handles preprocessing and feature engineering for customer data before applying clustering algorithms.

## Setup Instructions

To run the project, ensure you have the necessary dependencies. You can install them using the provided `requirements` file.

### 1. Clone the Repository

If you haven’t already, clone the repository using the following command:

```bash
git clone https://github.com/debjit721212/trimester1/apexproject.git
cd trimester1/apexproject
2. Install Dependencies
Install the required Python libraries by running:

bash
Copy code
pip install -r requirements_cluster_busters.txt
This will install all the necessary dependencies for running the notebooks and scripts.

3. Running the Notebooks
You can open and run each notebook using Jupyter Notebook or JupyterLab:

bash
Copy code
jupyter notebook
Alternatively, you can run each notebook directly from the command line using the nbconvert utility.

bash
Copy code
jupyter nbconvert --to notebook --execute <notebook_name>.ipynb
4. Dataset
The project uses customer data, which is available as a processed version in the repository. The data_extraction_instacart.ipynb notebook handles the extraction of the dataset, so make sure you run it first to have the required data available.

5. Data Files
Some of the output files generated during clustering are saved in the following formats:

hdbscan_clusters_all.parquet

hdbscan_clusters_test.parquet

hier_clusters_all.parquet

hier_clusters_test.parquet

kmeans_clusters_all.parquet

kmeans_clusters_test.parquet

These files are used to save the results of the clustering algorithms.

Clustering Techniques Used
1. K-Means Clustering
K-Means is a popular clustering technique where the data is divided into K clusters. The model assigns each customer to a cluster based on the nearest centroid.

Algorithm Steps:

Randomly initialize K centroids.

Assign each data point to the nearest centroid.

Recompute centroids as the mean of the points assigned to each centroid.

Repeat steps 2-3 until convergence (i.e., centroids don’t change).

2. Hierarchical Clustering
Hierarchical Clustering builds a tree-like structure of clusters called a dendrogram. It can either be agglomerative (bottom-up) or divisive (top-down).

Algorithm Steps:

Start with each data point as its own cluster.

Merge the two closest clusters.

Repeat the process until all points belong to a single cluster.

3. HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
HDBSCAN is an extension of DBSCAN, which is a density-based clustering algorithm. It handles noise and irregularly shaped clusters better than K-Means and hierarchical clustering.

Algorithm Steps:

Build a hierarchy of clusters based on the density of points.

Extract the most stable clusters from the hierarchy.

4. UMAP (Uniform Manifold Approximation and Projection)
UMAP is used for dimensionality reduction before clustering. It helps in reducing the high-dimensional data to a 2D or 3D space for easier visualization and understanding.

UMAP + HDBSCAN: The reduced dimensions from UMAP are passed to HDBSCAN to identify clusters.

Evaluation Metrics
The clustering results are evaluated using Recall@20 (for recommendation systems), Silhouette Score, and Adjusted Rand Index (ARI) to assess the quality of clustering.

1. Silhouette Score:
Measures how similar a point is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering.

2. Adjusted Rand Index (ARI):
Measures the similarity between two clustering results while correcting for chance. Higher ARI indicates a better match between clustering results.

Visualization and Storytelling
Once the clustering is complete, visualizations are generated to show the customer segments in a 2D space (after dimensionality reduction via UMAP) and the relationship between the clusters.

Storytelling:
Use the clusters to identify patterns in customer behavior (e.g., high-value customers, frequent buyers).

The goal is to interpret and visualize customer segments to make business decisions (such as targeted marketing).

Presentation
The PPT for the project presentation is included in the repository as both PDF and PPTX formats (Apex-Project-TeamPPT.pdf and Apex-Project-TeamPPT.pptx).

Conclusion
This project demonstrates the end-to-end process of customer segmentation using clustering algorithms. It involves data extraction, preprocessing, clustering, and visualization to understand customer behavior and make decisions based on the clustering results.

Feel free to explore the notebooks and adapt the project for your use cases!
