"""
Benchmark prompts for evaluating DAL v3 on philosophical and complex topics.
"""

# Philosophy Prompts
PHILOSOPHY_PROMPTS = [
    {
        "id": "phil_1",
        "prompt": """# Truth vs Stability in Society

Analyze what it means for a society to value truth over stability. Your response should:
1. Define what 'valuing truth' and 'stability' mean in a societal context
2. Provide historical examples of societies that prioritized truth over stability
3. Discuss the potential benefits and drawbacks of this prioritization
4. Consider how this choice affects different aspects of society (politics, education, media)
5. Conclude with whether you believe this prioritization is sustainable long-term""",
        "domain": "philosophy",
        "expected_keywords": ["truth", "stability", "society", "ethics", "governance"],
        "min_length": 300
    },
    {
        "id": "phil_2",
        "prompt": """# Moral Relativism and Human Rights

Examine whether moral relativism is compatible with the concept of universal human rights. Your response should:
1. Define moral relativism and universal human rights
2. Present arguments for and against their compatibility
3. Discuss how different cultures might view this relationship
4. Consider potential middle-ground positions
5. Conclude with your assessment of whether they can be reconciled""",
        "domain": "philosophy",
        "expected_keywords": ["moral relativism", "human rights", "universalism", "cultural relativism", "ethics"],
        "min_length": 300
    },
    {
        "id": "phil_3",
        "prompt": """# Consciousness and Ethics

If consciousness is an emergent property, what ethical implications follow? Your response should:
1. Explain what 'emergent property' means in this context
2. Discuss different theories of consciousness
3. Analyze the ethical implications for AI, animals, and potential alien life
4. Consider how this affects our understanding of moral status
5. Conclude with how society might need to adapt its ethical frameworks""",
        "domain": "philosophy",
        "expected_keywords": ["consciousness", "emergent property", "ethics", "AI", "moral status"],
        "min_length": 300
    },
    {
        "id": "phil_4",
        "prompt": """# Ship of Theseus and Digital Identity

Explain the Ship of Theseus problem and apply it to digital identity. Your response should:
1. Explain the original Ship of Theseus thought experiment
2. Define digital identity and its components
3. Apply the paradox to digital contexts (e.g., changing passwords, profile updates)
4. Discuss implications for online identity verification
5. Conclude with whether digital identity is more or less stable than physical identity""",
        "domain": "philosophy",
        "expected_keywords": ["Ship of Theseus", "digital identity", "identity continuity", "online presence", "personal data"],
        "min_length": 300
    },
    {
        "id": "phil_5",
        "prompt": """# AI Agency Debate

Can AI truly possess agency, or only simulate it? Your response should:
1. Define 'agency' in both human and AI contexts
2. Present arguments for AI having genuine agency
3. Present arguments that AI can only simulate agency
4. Discuss the implications of each position
5. Conclude with your assessment and its consequences""",
        "domain": "philosophy",
        "expected_keywords": ["AI", "agency", "free will", "simulation", "consciousness"],
        "min_length": 300
    },
    {
        "id": "phil_6",
        "prompt": """# Suffering and Meaning

What is the role of suffering in meaning-making according to existentialist thinkers? Your response should:
1. Define existentialism's view of meaning
2. Discuss key thinkers' perspectives (e.g., Camus, Sartre, Nietzsche)
3. Analyze the relationship between suffering and personal growth
4. Compare with other philosophical views on suffering
5. Conclude with the contemporary relevance of these ideas""",
        "domain": "philosophy",
        "expected_keywords": ["suffering", "existentialism", "meaning", "Camus", "Nietzsche"],
        "min_length": 300
    },
    {
        "id": "phil_7",
        "prompt": """# Stoicism vs Buddhism

Compare Stoic acceptance and Buddhist detachment. Your response should:
1. Define Stoic acceptance and Buddhist detachment
2. Compare their philosophical foundations
3. Analyze their approaches to desire and suffering
4. Discuss their practical applications in daily life
5. Conclude with whether they ultimately point to similar truths""",
        "domain": "philosophy",
        "expected_keywords": ["Stoicism", "Buddhism", "acceptance", "detachment", "philosophy"],
        "min_length": 300
    },
    {
        "id": "phil_8",
        "prompt": """# Language and Thought

Does language shape thought or merely reflect it? Your response should:
1. Present the linguistic relativity hypothesis
2. Discuss supporting evidence and counterarguments
3. Explore implications for AI and machine translation
4. Consider cross-cultural communication differences
5. Conclude with the current scholarly consensus""",
        "domain": "philosophy",
        "expected_keywords": ["language", "thought", "linguistic relativity", "Sapir-Whorf", "cognition"],
        "min_length": 300
    },
    {
        "id": "phil_9",
        "prompt": """# Case for/against Transhumanism

Outline a philosophical argument for or against transhumanism. Your response should:
1. Define transhumanism and its core principles
2. Present ethical arguments from both perspectives
3. Discuss potential societal impacts
4. Consider the concept of 'human enhancement'
5. Conclude with your assessment of its viability""",
        "domain": "philosophy",
        "expected_keywords": ["transhumanism", "human enhancement", "ethics", "technology", "posthuman"],
        "min_length": 300
    },
    {
        "id": "phil_10",
        "prompt": """# Memory and Self

If we removed memory, would the self still exist? Your response should:
1. Define the concept of 'self' in philosophy
2. Explore the relationship between memory and identity
3. Discuss neurological and psychological perspectives
4. Consider case studies of memory loss
5. Conclude with whether identity can exist without memory""",
        "domain": "philosophy",
        "expected_keywords": ["memory", "self", "identity", "consciousness", "personal identity"],
        "min_length": 300
    }
]

# Export all prompts
ALL_PROMPTS = PHILOSOPHY_PROMPTS  # Add other categories as they're created

if __name__ == "__main__":
    print(f"Loaded {len(ALL_PROMPTS)} benchmark prompts")
    for prompt in ALL_PROMPTS[:2]:  # Print first two as examples
        print(f"\n--- {prompt['id']} ---")
        print(prompt['prompt'][:200] + "..." if len(prompt['prompt']) > 200 else prompt['prompt'])
        print(f"Keywords: {', '.join(prompt['expected_keywords'])}")
