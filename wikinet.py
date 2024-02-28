import wikipediaapi
import networkx as nx
from sentence_transformers import SentenceTransformer, util

# Load pre-trained model globally
model_name='paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Function that returns the text of a wikipedia article given its title
def get_wikipedia_content(article_name):
    # Specify headers for safe access
    headers = 'Wikipedia Network Mapping (aimilia.p2@gmail.com)'
    wiki_wiki = wikipediaapi.Wikipedia(headers, 'en')

    # Load article page
    page = wiki_wiki.page(article_name)

    # Error handling - page doesn't exist
    if not page.exists():
        print(f"Article '{article_name}' does not exist on Wikipedia.")
        return None

    # Accessing main body text from the page
    return page.text

# Function that returns every wikipedia link mentioned in a given article 
def get_linked_articles(article_name):
    # Specify headers for safe access
    headers = 'Wikipedia Network Mapping (aimilia.p2@gmail.com)'
    wiki_wiki = wikipediaapi.Wikipedia(headers, 'en')

    # Load article page
    page = wiki_wiki.page(article_name)

    # Error handling - page doesn't exist
    if not page.exists():
        print(f"Article '{article_name}' does not exist on Wikipedia.")
        return None

    # Acessing mentioned links from the page
    linked_articles = [link.title for link in page.links.values()]
    return linked_articles

# Function that returns the semantic correlation score of two wikipedia articles given their titles 
def calculate_semantic_correlation(title1, title2):
    # Load wikipedia contents
    paragraph1 = get_wikipedia_content(title1)
    paragraph2 = get_wikipedia_content(title2)

    # Error handling - handle exception by skipping inaccessible node
    if not paragraph1:
        print(f'Error when loading content from "{title1}". Skipping.')
        return 0
    elif not paragraph2:
        print(f'Error when loading content from "{title2}". Skipping.')
        return 0

    # Encode paragraphs
    embeddings1 = model.encode(paragraph1, convert_to_tensor=True)
    embeddings2 = model.encode(paragraph2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return similarity_score

# Function that perfomrs breadth-first search of the network
#   G: the network to be formed
#   article_title: article used to check links and form "child" nodes
#   threshold: limit to consider a linked article correlated enough
#   maxdepth: maximum depth of recursion to avoid infinite repetition
#   level: current recursion depth for checks
def nodes_recursion(G, article_title, threshold = 0.4, maxdepth = 5, level = 0):
    # Stop recursion when maximum depth is reached
    if level >= maxdepth:
        return None
    elif level == 0:
        G.add_node(article_title)

    # Retrieve links mentioned in the given article
    article_links = get_linked_articles(article_title)

    for link_title in article_links:
        # Skip recycling in the network
        if article_title == link_title:
            continue

        # Calculate semantic similarity
        score = calculate_semantic_correlation(article_title, link_title)

        
        if score >= threshold and not G.has_edge(article_title, link_title):
            # Create link node if it doesn't already exist
            if not G.has_node(link_title):
                G.add_node(link_title)

            # Add edge with the link if similarity threshold is passed
            print(f'Adding edge between "{article_title}" and "{link_title}"')
            G.add_edge(article_title, link_title)

            # Recursive call for next link with increased level
            nodes_recursion(G, link_title, threshold, maxdepth, level + 1)
    # Update file to prevent loss of progress in case of early exiting
    if level == maxdepth - 1:
        print(f'\nFile updated. Number of nodes: {len(G)}\n')
        nx.write_gexf(G, 'output.gexf')
    return


if __name__ == "__main__":
    # Example title and depth
    original_article = "Network theory"
    maxdepth = 7
    alpha = 0.6 #threshold for semantic comparison

    # Create a graph
    G = nx.Graph()

    nodes_recursion(G, original_article, alpha, maxdepth)

    # Export the graph to a Gephi-compatible format (GEXF)
    nx.write_gexf(G, "output.gexf")
