import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for better compatibility
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import numpy as np


class NLPIllustrator:
    def __init__(self, save_path="images/generated", base_path="Lesson_1/images"):
        self.save_path = save_path
        self.base_path = base_path
        self._setup()
    
    def _setup(self):
        """Set up the illustration environment (e.g., creating necessary directories)."""
        os.makedirs(self.save_path, exist_ok=True)

        
    def illustrate_tokenization(self):
        """Create illustrations showing different tokenization methods."""
        # Example sentence to tokenize
        sentence = "L'intelligence artificielle transforme notre monde."
        
        # 1. Character-level tokenization - Create a separate figure just for this
        plt.figure(figsize=(10, 6))
        
        # Create a shorter alphabet-to-ID mapping (just the first 5 letters + "...")
        alphabet = list("abcde")
        char_ids = list(range(0, len(alphabet)))  # IDs starting from 0
        
        # Add title above the table, with padding
        plt.text(0.5, 0.9, "Encodage des caractères", 
                 horizontalalignment='center', 
                 fontsize=14, 
                 fontweight='bold')
        
        # Create a mapping table for character tokenization
        char_data = []
        for char, id_ in zip(alphabet, char_ids):
            char_data.append([char, id_])
        # Add ellipsis row
        char_data.append(["...", "..."])
        
        # Create table for character tokenization with alphabet mapping
        table = plt.table(
            cellText=char_data,
            colLabels=["Caractère", "ID"],
            loc="center",
            cellLoc="center",
            colWidths=[0.15, 0.15]
        )
        
        # Adjust font size for better readability
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/char_tokenization.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 2. Word and subword tokenization - Create a separate figure for these two
        plt.figure(figsize=(14, 10))
        
        # Word-level tokenization subplot
        plt.subplot(2, 1, 1)
        word_tokens = sentence.split()
        word_ids = list(range(1, len(word_tokens) + 1))
        
        # Create a mapping table for word tokenization
        word_data = []
        for word, id_ in zip(word_tokens, word_ids):
            word_data.append([word, id_])
        
        # Create table for word tokenization
        table = plt.table(
            cellText=word_data,
            colLabels=["Mot", "ID"],
            loc="center",
            cellLoc="center",
            colWidths=[0.3, 0.1]
        )
        
        # Adjust font size for better readability
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.title("Tokenisation par mot")
        plt.axis("off")
        
        # 3. Subword tokenization (BPE-like)
        plt.subplot(2, 1, 2)
        # Simulate BPE tokenization
        subword_tokens = ["L'", "intel", "ligence", "arti", "fici", "elle", "trans", "forme", "notre", "monde", "."]
        subword_ids = list(range(1, len(subword_tokens) + 1))
        
        # Create a mapping table for subword tokenization
        subword_data = []
        for subword, id_ in zip(subword_tokens, subword_ids):
            subword_data.append([subword, id_])
        
        # Create table for subword tokenization
        table = plt.table(
            cellText=subword_data,
            colLabels=["Sous-mot", "ID"],
            loc="center",
            cellLoc="center",
            colWidths=[0.2, 0.1]
        )
        
        # Adjust font size for better readability
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.title("Tokenisation par sous-mots (BPE)")
        plt.axis("off")
        
        plt.tight_layout(h_pad=4.0)
        plt.savefig(f"{self.save_path}/word_subword_tokenization.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Create a visualization of the BPE algorithm
        self.illustrate_bpe_algorithm()
    
    def illustrate_bpe_algorithm(self):
        """Illustrate how BPE (Byte Pair Encoding) algorithm works."""
        plt.figure(figsize=(14, 8))
        
        # Example vocabulary building with BPE
        words = ["lower", "lowest", "newer", "wider", "lower"]
        
        # BPE merges
        merges = [
            ("l", "o", "lo"),
            ("lo", "w", "low"),
            ("e", "r", "er"),
            ("low", "er", "lower"),
            ("low", "e", "lowe"),
            ("s", "t", "st")
        ]
        
        # Create a visualization of the BPE process
        plt.subplot(2, 1, 1)
        
        # Show initial tokenization
        initial_tokens = []
        for word in words:
            initial_tokens.append(" ".join(list(word)))
        
        y_positions = np.arange(len(words))
        plt.barh(y_positions, [0.1] * len(words), height=0.4, left=0, color='white', alpha=0)
        
        # Add text labels for words and their character tokenization
        for i, (word, tokens) in enumerate(zip(words, initial_tokens)):
            plt.text(-0.15, i, f"{word}:", ha="right", va="center", fontweight="bold")
            plt.text(0.1, i, tokens, ha="left", va="center")
        
        plt.yticks([])
        plt.xticks([])
        plt.xlim(-2, 15)
        plt.title("Tokenisation initiale par caractère")
        
        # Show BPE merges
        plt.subplot(2, 1, 2)
        
        # Create a table to show the BPE merges
        merge_data = []
        for i, (token1, token2, merged) in enumerate(merges):
            merge_data.append([i+1, f"{token1} + {token2}", merged])
        
        plt.table(
            cellText=merge_data,
            colLabels=["Étape", "Fusion", "Nouveau token"],
            loc="center",
            cellLoc="center",
            colWidths=[0.1, 0.2, 0.2]
        )
        plt.title("Processus de fusion BPE")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/bpe_algorithm.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Create a visualization of vocabulary size comparison
        self.illustrate_vocab_size_comparison()
    
    def illustrate_vocab_size_comparison(self):
        """Illustrate the vocabulary size comparison between different tokenization methods."""
        plt.figure(figsize=(12, 6))
        
        # Approximate vocabulary sizes
        vocab_sizes = {
            "Caractères": 128,  # ASCII
            "Sous-mots (BPE)": 30000,  # Typical subword vocab
            "Mots": 500000,  # Typical word vocab for English
        }
        
        # Create bar chart
        methods = list(vocab_sizes.keys())
        sizes = list(vocab_sizes.values())
        
        plt.bar(methods, sizes, color=['#3498db', '#2ecc71', '#e74c3c'])
        plt.yscale('log')
        plt.ylabel('Taille du vocabulaire (échelle log)')
        plt.title('Comparaison des tailles de vocabulaire')
        
        # Add value labels on top of bars
        for i, v in enumerate(sizes):
            if v >= 1000:
                plt.text(i, v*1.1, f"{v/1000:.0f}K", ha='center')
            else:
                plt.text(i, v*1.1, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/vocab_size_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    def illustrate_sentiment_analysis(self):
        """Create an illustration showing sentiment analysis on reviews."""
        # Create mock movie reviews with sentiments
        reviews = [
            "This movie was absolutely fantastic! Great acting and storytelling.",
            "Worst film I've seen in years. Complete waste of time and money.",
            "Decent movie but the ending was disappointing.",
            "I loved everything about this film, can't wait for the sequel!",
            "Boring plot and terrible dialog. Couldn't finish watching it."
        ]
        
        # Create a mock sentiment analysis visualization
        sentiments = [0.9, -0.8, 0.2, 0.95, -0.7]  # Positive to negative scale
        
        plt.figure(figsize=(12, 6))
        colors = ['green' if s > 0 else 'red' for s in sentiments]
        
        plt.barh(range(len(reviews)), sentiments, color=colors)
        plt.yticks(range(len(reviews)), [r[:40] + "..." for r in reviews])
        plt.xlabel('Sentiment Score (Negative to Positive)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Sentiment Analysis on Movie Reviews')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/sentiment_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    def illustrate_translation(self):
        """Create an illustration of machine translation."""
        # Example sentences in different languages
        languages = ['English', 'French', 'Spanish', 'German', 'Chinese']
        sentences = {
            'English': "Hello, how are you today?",
            'French': "Bonjour, comment allez-vous aujourd'hui?",
            'Spanish': "Hola, ¿cómo estás hoy?",
            'German': "Hallo, wie geht es Ihnen heute?",
            'Chinese': "你好，今天怎么样？"
        }
        
        # Create a visual representation of translation
        plt.figure(figsize=(10, 6))
        
        # Draw a network-like structure showing translations
        pos = {
            'English': (0, 0),
            'French': (-1, 1),
            'Spanish': (1, 1),
            'German': (-1, -1),
            'Chinese': (1, -1)
        }
        
        # Plot nodes (languages)
        for lang, (x, y) in pos.items():
            plt.plot(x, y, 'o', markersize=15, label=lang)
            plt.text(x, y+0.1, lang, ha='center')
            plt.text(x, y-0.3, sentences[lang], ha='center', fontsize=8)
        
        # Plot edges (translations)
        for lang1 in languages:
            for lang2 in languages:
                if lang1 != lang2:
                    plt.plot([pos[lang1][0], pos[lang2][0]], 
                             [pos[lang1][1], pos[lang2][1]], 
                             'k-', alpha=0.2)
        
        plt.title('Machine Translation Connects Languages')
        plt.axis('equal')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/translation.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def illustrate_llm_future(self):
        """Create an illustration showing the future of LLMs with advanced reasoning and agent systems."""
        # Create a figure showing LLM capabilities evolution
        plt.figure(figsize=(12, 8))
        
        # 1. Create data for model scaling
        model_sizes = [0.1, 0.5, 1, 2, 5, 10, 20, 70, 170, 540]  # Billion parameters
        reasoning_capabilities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.75, 0.85, 0.95]  # Hypothetical reasoning scores
        
        # Define the emergent reasoning threshold
        threshold = 0.7
        
        # Plot model scaling curve
        plt.subplot(2, 2, 1)
        plt.plot(model_sizes, reasoning_capabilities, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Émergence du raisonnement')
        plt.xscale('log')
        plt.xlabel('Taille du modèle (milliards de paramètres)')
        plt.ylabel('Capacité de raisonnement')
        plt.title('Émergence des capacités avec la taille')
        plt.grid(alpha=0.3)
        plt.legend()
        
        # 2. Create a visualization of test-time scaling
        plt.subplot(2, 2, 2)
        
        # Sample data showing improved performance with test-time computation
        inference_steps = [1, 2, 4, 8, 16, 32, 64]
        accuracy = [65, 68, 72, 78, 85, 90, 94]
        
        plt.plot(inference_steps, accuracy, 'o-', color='green', linewidth=2, markersize=8)
        plt.xlabel('Nombre d\'étapes d\'inférence')
        plt.ylabel('Précision (%)')
        plt.title('Test-Time Scaling')
        plt.grid(alpha=0.3)
        
        # 3. Create a network visualization for multi-agent systems
        plt.subplot(2, 1, 2)
        
        # Create a graph representing agents and their interactions
        G = nx.DiGraph()
        
        # Add nodes representing different specialized agents
        agents = [
            "Agent planificateur", 
            "Agent recherche", 
            "Agent critique", 
            "Agent mémoire",
            "Agent créatif",
            "Agent exécution"
        ]
        
        # Add nodes
        for agent in agents:
            G.add_node(agent)
        
        # Add edges representing communication and task flow
        G.add_edge("Agent planificateur", "Agent recherche")
        G.add_edge("Agent planificateur", "Agent créatif")
        G.add_edge("Agent recherche", "Agent mémoire")
        G.add_edge("Agent créatif", "Agent critique")
        G.add_edge("Agent recherche", "Agent critique")
        G.add_edge("Agent mémoire", "Agent planificateur")
        G.add_edge("Agent mémoire", "Agent créatif")
        G.add_edge("Agent critique", "Agent exécution")
        G.add_edge("Agent critique", "Agent planificateur")
        G.add_edge("Agent exécution", "Agent mémoire")
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=1800, node_color='lightblue', alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        nx.draw_networkx_edges(
            G, pos, width=1.5, alpha=0.7, 
            edge_color='gray', 
            connectionstyle='arc3,rad=0.1',
            arrowsize=15
        )
        
        plt.title('Système multi-agents pour le raisonnement complexe')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/llm_future.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def generate_all(self):
        """Call all defined illustration functions."""
        illustration_methods = [method for method in dir(self) if method.startswith("illustrate_")]
        for method in tqdm(illustration_methods, desc="Generating illustrations",
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
            print(f"Generating: {method}")
            getattr(self, method)()


if __name__ == "__main__":
    illustrator = NLPIllustrator()
    illustrator.generate_all()
