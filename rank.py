from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class WebPage:
    def __init__(self, url, content, outbound_links=[]):
        self.url = url
        self.content = content
        self.outbound_links = outbound_links
        self.inbound_links = []

def load_web_pages(C:\Users\harsh\OneDrive - Amrita Vishwa Vidyapeetham\2nd SEM CLASS\MFC_2):
    web_pages = []
    for path in C:\Users\harsh\OneDrive - Amrita Vishwa Vidyapeetham\2nd SEM CLASS\MFC 2:
        with open(path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text(separator=' ')
            outbound_links = [a['href'] for a in soup.find_all('a', href=True)]
            web_pages.append(WebPage(path, text_content, outbound_links))
    return web_pages

def calculate_relevance(query, web_pages):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + [page.content for page in web_pages])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    content_similarity = similarity_matrix[0, 1:]

    relevance_scores = []
    for i, page in enumerate(web_pages):
        content_score = content_similarity[i]
        relevance_scores.append(content_score)

    return relevance_scores

def create_link_matrix(web_pages):
    size = len(web_pages)
    M = np.zeros((size, size))

    url_to_index = {page.url: index for index, page in enumerate(web_pages)}

    for i, page in enumerate(web_pages):
        if page.outbound_links:
            for link in page.outbound_links:
                j = url_to_index.get(link)
                if j is not None:
                    M[j][i] = 1.0 / len(page.outbound_links)

    return M

def page_rank(M, beta=0.8, threshold=1e-10):
    N = M.shape[0]
    r = np.ones(N) / N
    c = (1 - beta) / N

    for _ in range(100):
        r_new = beta * np.dot(M, r) + c
        if np.linalg.norm(r_new - r) < threshold:
            break
        r = r_new

    return r

def main():
    file_paths = [
        "path/to/index.html",
        "path/to/page2.html",
        "path/to/page3.html",
        "path/to/page4.html"
    ]

    query = input("Enter your query: ")
    web_pages = load_web_pages(file_paths)
    relevance_scores = calculate_relevance(query, web_pages)

    link_matrix = create_link_matrix(web_pages)
    authority_scores = page_rank(link_matrix)

    combined_scores = [0.5 * relevance + 0.5 * authority for relevance, authority in zip(relevance_scores, authority_scores)]
    sorted_pages = sorted(zip(web_pages, combined_scores), key=lambda x: x[1], reverse=True)

    for rank, (page, score) in enumerate(sorted_pages, start=1):
        print(f"Rank {rank}: {page.url} - Score: {score}")

if __name__ == "__main__":
    main()
