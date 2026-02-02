# XAI: A Comparative Study of Post-hoc and Intrinsic Explainability Methods for Categorical Tabular Data.

## Context
This Master’s thesis focuses on eXplainable Artificial Intelligence (XAI) and aims to compare several explainability methods within the framework of a multi-class prediction task applied to an almost fully categorical dataset.
The dataset consists of a survey on the habits of Quebec residents conducted in 2022. One of the primary objectives
of collecting this dataset was to enable the development of a tool allowing individuals to position themselves on
the political spectrum solely based on their lifestyle and demographic features (e.g., age, income). This represents
a challenging and relatively uncommon case on which to test both the explainability methods and the comparison
benchmark on a real-world dataset.  

## Challenges
A central challenge of this task lies in defining criteria that allow for a meaningful comparison between fundamentally different explainability techniques. This difficulty arises from several factors, including diversity in
explanation techniques, variability in explanation outputs, dependence on data type, and the lack of consensus
on what constitutes a good explanation in the Machine Learning (ML) field. To address this challenge, a structured comparison framework is proposed in Section 5, largely inspired by the functionally grounded evaluation
approaches described in [14] and [35]. Those scientific articles provide answers to the question of how to evaluate
and compare XAI methods in a systematic way, based on the Ethics Guidelines for Trustworthy AI published by
the European Union’s High Level Expert Group (AI HLEG) [20]. 


## Thesis structure
This work begins by introducing XAI and the accuracy and interpretability tradeoff in section 1.2. Section 2
presents the datasets and the predictive model, a Neural Network (NN), used as a black-box model to be explained.
Section 3 defines the theoretical foundation of the three post-hoc explainability techniques used — **SHAP, LIME
and Counterfactuals** — while Section 4 focuses on the intrinsically interpretable model used — **Optimal Sparse
Decision Tree** (OSDT). Those techniques were chosen because they suit the nature of the data and the task at
hand. Moreover, the three post-hoc techniques are commonly used and I was interested in understanding them
more deeply and to see how they compare to an intrinsically interpretable model and to each other’s. The section
5 defines the characteristics providing the formal definition of a XAI’s method desired properties, on which we
are going to compare the 4 approaches. The section 6 discusses the findings and concludes the thesis.


[If you require the full work send me a message at simeon7730@proton.me]
