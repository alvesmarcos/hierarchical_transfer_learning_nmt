# Changelog 

A cada treinamento geral da rede é adicionado uma versão com os resultados parciais avaliando o treinamento gerado.

## [0.0.1] - 2020-06-08

### Descrição

Primeiro teste geral realizado utilizando todas as técnicas desenvolvidas para incorporar novas palavras ao vocabulário inicial.

### Dados gerais
1. **Treinamentos** - 16 treinamentos entre os dias <ins>04 e 08 de Junho</ins>.
2. **Arquitetura** - [Pay Less Attention with Lightweight and Dynamic Convolutions](https://github.com/pytorch/fairseq/blob/master/examples/pay_less_attention_paper/README.md)
3. **Parâmetros** - [1](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/master/params/train.json) & [2](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/cae2cb0ae8c2039832511bce839c936ef0fc9f9a/params/train.json)
4. **Estratégias** - [Padrão](#padrao), [Randômica](randomica), Similaridade e GloVe
5. **Divisão de Dados** - 50%, 75% e 100%

### Treinamentos

Para rodar cada treinamento foi utilizado a ferramenta [Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) desenvolvida pela Google.

#### Padrão

*[Parâmetros 1](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/master/params/train.json)*

* 50% - ~Informar dados dos dicionários e sentenças~
<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__fairseq__50@2020-06-03.13.40.14.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__fairseq__50@2020-06-03.13.40.14.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 75% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__fairseq__75@2020-06-04.21.37.08.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__fairseq__75@2020-06-04.21.37.08.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 100% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__fairseq__100@2020-06-04.21.15.10.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__fairseq__100@2020-06-04.21.15.10.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

*[Parâmetros 2](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/cae2cb0ae8c2039832511bce839c936ef0fc9f9a/params/train.json)*

* 50% - ~Informar dados dos dicionários e sentenças~
<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__fairseq__50@2020-06-06.18.04.01.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__fairseq__50@2020-06-06.18.04.01.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 75% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__fairseq__75@2020-06-06.18.16.05.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__fairseq__75@2020-06-06.18.16.05.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 100% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__fairseq__100@2020-06-06.18.25.01.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__fairseq__100@2020-06-06.18.25.01.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

#### Randômica

*[Parâmetros 1](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/master/params/train.json)*

* 75% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__randomly__75@2020-06-04.21.30.59.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__randomly__75@2020-06-04.21.30.59.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 100% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__randomly__100@2020-06-05.00.29.33.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__randomly__100@2020-06-05.00.29.33.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

*[Parâmetros 2](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/cae2cb0ae8c2039832511bce839c936ef0fc9f9a/params/train.json)*

* 75% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__randomly__75@2020-06-07.01.55.55.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__randomly__75@2020-06-07.01.55.55.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 100% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__randomly__100@2020-06-08.09.13.09.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__randomly__100@2020-06-08.09.13.09.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

#### Similaridade

*[Parâmetros 1](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/master/params/train.json)*

* 75% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__similarity__75@2020-06-05.23.14.14.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__similarity__75@2020-06-05.23.14.14.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 100% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__similarity__100@2020-06-06.00.32.38.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__similarity__100@2020-06-06.00.32.38.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

*[Parâmetros 2](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/cae2cb0ae8c2039832511bce839c936ef0fc9f9a/params/train.json)*

* 75% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__similarity__75@2020-06-07.04.16.29.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__similarity__75@2020-06-07.04.16.29.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 100% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__similarity__100@2020-06-08.05.04.14.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__similarity__100@2020-06-08.05.04.14.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

#### GloVe

*[Parâmetros 2](https://github.com/alvesmarcos/hierarchical_transfer_learning_nmt/blob/cae2cb0ae8c2039832511bce839c936ef0fc9f9a/params/train.json)*

* 75% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__glove300_pretrained__100@2020-06-08.11.48.06.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__glove300_pretrained__100@2020-06-08.11.48.06.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~

* 100% - ~Informar dados dos dicionários e sentenças~

<p float="left">
  <img src="resources/best_loss@train__lightconv_iwslt_de_en__glove300_pretrained__100@2020-06-08.11.48.06.png" width="300" />
  <img src="resources/loss@train__lightconv_iwslt_de_en__glove300_pretrained__100@2020-06-08.11.48.06.png" width="300" />
</p>

**BLEU**: ~Informar Bleu~


[Unreleased]: https://github.com/zokla-io/MaisOpcao/tree/dev
[0.0.1]: https://github.com/zokla-io/MaisOpcao/tree/dev
