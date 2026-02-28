This project was my exam submission for Advanced Topics in NLP course at UCPH. Here, I explore how the Transformer architecture compares to older architectures, such as RNNs, in terms of systematic generalization. My work is a reproduction of the experiments presented in the paper _Generalization without Systematicity_, which originally investigated the degree to which RNNs can generalize to new or rarely seen words in novel contexts.

The primary motivation behind my project is to test the hypothesis that the Transformer architecture will generally outperform RNN networks on these tasks. I also sought to evaluate whether a pretrained model like BART, despite its power, might be hindered by the specific, limited vocabulary of the target dataset compared to a standard "vanilla" Transformer.


### Methodology and Architectures

To conduct this investigation, I employed two distinct Transformer-based approaches:

-   **Vanilla Models**: These are standard Transformer architectures implemented from scratch without any pretraining on data outside the scope of the experiments.
    
    
-   **BART Models**: I utilized the standard BART architecture with weights pretrained by Facebook, which I then fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) methods. This included the use of a language head and three different adapters: **LoRA blocks**, **IA3**, and **bottleneck adapters**.

    

For the BART models, I introduced a novel approach to the language head to avoid the memory overhead of BART's standard 50,000-token vocabulary. I created a smaller, specialized vocabulary consisting only of the byte-wise tokens needed for the specific dataset, which significantly reduced the dimensions of the language head.


### Dataset and Experimental Design

I utilized the **SCAN dataset**, which consists of 20,000 fully labeled input and target sequence pairs. The dataset is structured so that input sequences combine primitive commands (like "turn left") with modifiers (like "twice"), which translate into sequences of output primitives. My project reproduces three specific experiments from the original study:


-   **Experiment 1 (E1)**: Testing generalization to random subsets of commands by training the models on varying percentages of the full dataset (from 1% to 64%).
    
-   **Experiment 2 (E2)**: Testing generalization to longer action sequences, where the training set contains sequences up to 22 actions, while the test set requires generating sequences up to 48 actions.
    
-   **Experiment 3 (E3)**: Testing compositional generalization across specific primitive commands, such as "turn left" and "jump," by varying the amount of exposure the model has to these primitives in complex contexts during training.
