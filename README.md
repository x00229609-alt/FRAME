# FRAME (Formal Reasoner and Model Explainer): A Framework for Democratising Formal Explainable Artificial Intelligence Techniques Across the AI Lifecycle

FRAME unifies several existing Formal XAI techniques into a single Python interface, enabling trustworthy, human-friendly explanations — tailored to the user's technical background — across a range of AI model architectures.

#### Model Architectures
The supported model architectures are:
- Random Forest
- Logistic Regression
- Deep Neural Networks
- XGBoost

With source code being extended and adapted from the following repositories:
[FoXplainer](https://github.com/trustablefox/foxplainer)
[VeriX](https://github.com/NeuralNetworkVerification/VeriX)
[XReason](https://github.com/alexeyignatiev/xreason)

<img width="1857" height="1296" alt="frame_300" src="https://github.com/user-attachments/assets/9917a25f-ac75-49c6-8c5d-b5c9da300a79" />


#### Explanations
All the repositories have been extended to support:
- Abductive explanations (AXPs, 'Why did a prediction happen?')
- Contrastive explanations (CXPs, 'What would make another prediction happen?')
- Formal Feature Attribution (FFA, 'Which features were most important?')

With explanations being presented differently according to one of three selected stakeholder types:
- **Developers** : those involved directly with model development and maintenance
    - FFA, 1 AXP, 2 CXPs  
- **Non-Technical Decision Makers** : those involved with making decisions in partnership with AI suggestions
    - 2 AXPs, 2 CXPs  
- **Model Subjects**: those impacted by the final decision influenced from the AI suggestion
    - LLM synthesis of 1 AXP and 2 CXPs​

These visualisations are represented in HTML within jupter notebooks. Some examples of the visualisations can be seen below:

**Decision Maker Visualisation**

<img width="800" height="318" alt="decisionmaker-readme" src="https://github.com/user-attachments/assets/682812a9-381f-43d5-a194-fdda44334767" />

**Developer Visualisation**

<img width="800" height="398" alt="developer-readme" src="https://github.com/user-attachments/assets/d048cbc2-3d68-4189-9e1e-143b836b06fc" />

**Model Subject Visualisation**

<img width="1487" height="341" alt="Screenshot 2026-08-27 at 14 49 32" src="https://github.com/user-attachments/assets/a3d31c89-e1f6-4053-9a0a-664fffa364bd" />
