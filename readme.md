

# Classificação Zernike Moments
Implementação de Zernike Moments para Extração de Características de Imagens

## Equipe
- Adeline Rodrigues Cruz Wyse Guimarães

## Descrição do(s) Descritor(es) Implementado
###  O Zernike Moments
O Zernike Moments é um descritors baseado em forma que captura características geométricas das imagens. Não varia com a rotação, escala nem translação, o que o torna útil para o reconhecimento de padrões. O Zernike Moments é calculado a partir de polinômios ortogonais e fornece uma representação precisa da forma dos objetos nas imagens.

## Repositório do Projeto
[Link do repositório do projeto](https://github.com/AdeWyse/classificacao)

## Video Apresentando o Projeto
[Link do video apresentando o projeto](https://drive.google.com/file/d/1STAJYJdgzy3BCnRBN2Bi1jLUKIJ6GeCZ/view?usp=drive_link)

## Classificador e Acurácia
- Multi-Layer Perceptron - 92.8%
- Random Forest - 91.0%
- Support Vector Machine - 85.7%


## Instruções de Uso
### Pré-requisitos
- Configuração de ambiente confirme apresentada na disciplina

### Instalação
1. Clonar o repositório:
    ```sh
    git clone https://github.com/AdeWyse/classificacao
    cd classificacao
    ```

3. Instalar as dependências:
    ```sh
    conda install -c conda-forge mahotas
    conda install numpy opencv scikit-learn progress
    ```

### Execução
Se a pasta /feature_labels/zernikemoments não existir rodar o arquivo zernikemonments_FeatureExtraction.py

Após rodar o arquivo a classificação pode ser feita utilizando um dos 3 classificadores abaixo:

- Multi-Layer Perceptron - mlp.classifier.py
- Random Forest - rf.classifier.py
- Support Vector Machine - svm.classifier.py

Ou todos eles pelo arquivo run_all_classifier.py .

