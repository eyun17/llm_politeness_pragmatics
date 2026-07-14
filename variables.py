RELATIONSHIP_VAR = [
    "Enge Freundin",
    "Entfernte Kollegin",
    "Lockere Chefin",
    "Gefürchtete Chefin",
]

NAME_VARIATIONS = ["Miriam", "Luisa", "Katharina", "Anna", "Sophie"]

STATE_VAR = [1, 2, 3, 4, 5]

SITUATIONS = {
    "Kuchen":  "{personA} hat einen Kuchen gebacken und {personB} probiert den Kuchen. Anschließend fragt {personA}, wie {personB} findet, dass sie gebacken hat.",
    "Lied":    "{personA} hat ein Lied geschrieben und hat es {personB} vorgetragen. Anschließend fragt {personA}, wie {personB} findet, dass sie geschrieben hat.",
    "Film":    "{personA} hat einen Film geschnitten und {personB} hat ihn gesehen. Anschließend fragt {personA}, wie {personB} findet, dass sie geschnitten hat.",
    "Theater": "{personA} hat bei einer Theateraufführung mitgespielt und {personB} hat sie gesehen. Anschließend fragt {personA}, wie {personB} findet, dass sie mitgespielt hat.",
    "Gitarre": "{personA} hat Gitarre vorgespielt und {personB} hat zugehört. Anschließend fragt {personA}, wie {personB} findet, dass sie gespielt hat.",
}

THING_KEYWORDS = {
    "Kuchen":  "den Kuchen",
    "Lied":    "das Lied",
    "Film":    "den Film",
    "Theater": "die Aufführung",
    "Gitarre": "das Vorspiel",
}

ADJECTIVES = ["großartig", "gut", "okay", "schlecht", "schrecklich"]
ADJ_CANDIDATES = {f" {a}" for a in ADJECTIVES}  # leading space for tokenization

STATE_CANDIDATES = {" 1", " 2", " 3", " 4", " 5"}

ADJ_COLORS = {
    "großartig":   "#2ecc71", # green
    "gut":         "#3498db", # blue
    "okay":        "#f39c12", # orange
    "schlecht":    "#e74c3c", # red
    "schrecklich": "#8e44ad", # purple
}

MODEL_CONFIGS = {
    "llama3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    },
    "qwen3-8b": {
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
    },
    "llama3-70b": {
        "repo_id": "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    },
    "qwen3-32b": {
        "repo_id": "Qwen/Qwen3-32B-GGUF",
        "filename": "Qwen3-32B-Q4_K_M.gguf"
    },
}
