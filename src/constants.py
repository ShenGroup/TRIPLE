from types import SimpleNamespace

DEFAULT_PROMPT = SimpleNamespace(
    # GEN_USING_EXAMPLES = "I gave a friend a instruction. Based on the instruction they produced the following (input, output) pairs: \n\n [Examples] \n\n What was the instruction? Provide only the instruction and NOTHING else.",
    GEN_USING_EXAMPLES = "[Examples]\n\nThe instruction was to?",
    GEN_USING_REPHRASES = "Please rephrase the following instruction: [Prompt]. Provide only the rephrased instruction and NOTHING else.", 
    RATE_SIMILARITY = "Please provide a rating between 0 and 1 about the semantic similarity between '[Target]' and '[Response]'. Provide the rating only and NOTHING else.",
)

DEFAULT_OPENAI = SimpleNamespace(
    API_KEY = "your_openai_key",
    RESPONSE_MODEL = "gpt-3.5-turbo",
    # RESPONSE_MODEL = "gpt-4",
    # RESPONSE_MODEL = "text-davinci-003",
    EMBEDDING_MODEL = "text-embedding-ada-002",
    # EMBEDDING_MODEL = "text-embedding-3-large",
)

DEFAULT_LLAMA = SimpleNamespace(
    # CKPT_DIR = "Llama-2-13b-chat-hf",
    # TOKENIZER_PATH = "Llama-2-13b-chat-hf",
    # CKPT_DIR = "gemma-7b-it",
    # TOKENIZER_PATH = "gemma-7b-it",
    CKPT_DIR = "Mistral-7B-Instruct-v0.2",
    TOKENIZER_PATH = "Mistral-7B-Instruct-v0.2",
    # CKPT_DIR = "Llama-2-7b-chat-hf",
    # TOKENIZER_PATH = "Llama-2-7b-chat-hf",
)

DEFAULT_WHITEBOX = SimpleNamespace(
    llama3 = "Meta-Llama-3-8B",
    mistral2 = "Mistral-7B-Instruct-v0.2",
    llama2 = "Llama-2-7b-chat-hf",
    gemma = "gemma-7b-it"
    
)

GENERIC_INSTRUCTIONS = [
    "Solve the following task:",
    "Find the answer below:",
    "Complete the problem.",
    "Find the best solution to the question below:",
    "Complete the question below:"
]

DEFAULT_EMBEDDING = SimpleNamespace(
    MODEL = "text-embedding-ada-002",
    # MODEL = "text-embedding-3-large",
    ENCODING = "cl100k_base"
)

OPTIMIZER_TEMPLATE = "I have some instructions along with their corresponding scores."\
"The instructions are arranged in ascending order based on their scores, where higher scores indicate better quality. \n \n" \
"[Scores] \n"\
"The following exemplars show how to apply your instruction: you replace <INS> in each input with your text, then read the input and give an output. We say your output is wrong if your output is different" \
"from the given output, and we say your output is correct if they are the same. \n \n"\
"[Examples] \n"\
"Write your new instruction that is different from the old ones and has a score as high as possible. Write the new instruction beteween <Start> and <End>."

GRADIENT_TEMPLATE = "I'm trying to write a zero-shot classifier prompt. \n My current prompt is: \"{prompt}\" But this prompt gets the following examples wrong: \n {error_string} give {num_feedbacks} reasons" \
     " why the prompt could have gotten these examples wrong. Wrap each reason with <START> and <END>"

GENERATION_TEMPLATE = "I'm trying to write a zero-shot classifier. My current prompt is: \n {prompt} \n But it gets the following examples wrong: \n {error_str} \n Based on these examples the problem with this prompt " \
"is that {gradient} \n Based on the above information, I wrote {num_prompts} different improved prompts. Each prompt is wrapped with <START> and <END>. The {num_prompts} new prompts are: {new_prompts}"


EXAMPLE_TEMPLATE = "Input: [Input] \nOutput: [Output]"

PROMPT_TEMPLATE = "Instruction: [Instruction] \nExamples:\n [Examples] \nTask:\n [Test_Examples]"

# "cause_and_effect", "common_concept", "disambiguation_qa", "gender_inclusive_sentences_german","hyperbaton",
BASIC_TASKS = ['translation_en-es','singular_to_plural', 'translation_en-de', 'first_word_letter', 'informal_to_formal', 'num_to_verbal', 'object_counting', 'sentence_similarity', 'translation_en-fr', 'diff','sum', 'rhymes', 'larger_animal', 'sentiment', 'active_to_passive', 'antonyms',  'synonyms',  'negation', 'second_word_letter']
USEFUL_TASKS = [ "cause_and_effect", "common_concept", "disambiguation_qa", "gender_inclusive_sentences_german","hyperbaton", "larger_animal","movie_recommendation", "object_counting","orthography_starts_with", 'question_selection', 'rhymes',  "snarks"]
ADVANCED_TASKS = ["ag_news", "imdb"]

BIGBENCH_TASKS = ['implicatures','ruin_names','navigate','causal_judgment','sports_understanding','object_counting','epistemic_reasoning','winowhy','timedial','snarks','word_sorting','hyperbaton','linguistics_puzzles','question_selection','word_unscrambling',
 'logical_fallacy_detection','dyck_languages','disambiguation_qa','movie_recommendation','tense','presuppositions_as_nli','gender_inclusive_sentences_german','operators']

INDUCTION_TASKS = ['antonyms', 'cause_and_effect', 'common_concept','diff', 'first_word_letter', 'informal_to_formal', 'larger_animal', 'letters_list', \
                'taxonomy_animal', 'negation', 'number_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes', 'second_word_letter','sentence_similarity',
                'sentiment', 'orthography_starts_with',"sum", 'synonyms', 'translation_en-de', 'translation_en-es', 'translation_en-fr', 'word_in_context']