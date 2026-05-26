# 📊 Homicídios Globais (UNODC) - EDA e Modelagem Preditiva

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

### 👥 Equipe de Desenvolvimento
* Antonio Acrisio Caxias Sousa
* Antonio Bryan de Almeida Abreu
* Eros Ryan Simette
* Fernando Anderson Borges
* Francisco Breno Gomes Melo
* Francisco Felipe Rodrigues de Sousa
* Paulo Ricardo Sousa Menezes

---

Repositório dedicado ao **Trabalho 01** da disciplina de Tópicos Especiais em Computação I (Prof. Iális Cavalcante). 

O projeto aplica técnicas avançadas de Ciência de Dados para analisar, limpar e modelar a base de dados oficial da **UNODC** (Escritório das Nações Unidas sobre Drogas e Crime) referente aos homicídios intencionais globais entre 2013 e 2022, com projeções estatísticas até 2026.

---

## 🛠️ Pipeline de Dados e Tratamento

Para garantir o rigor analítico e evitar distorções matemáticas presentes em datasets globais, aplicamos as seguintes etapas de engenharia de dados:
* **Filtros de Sanidade Estatística:** Identificação e remoção de *outliers* extremos (taxas > 150 por 100 mil hab.) causados por micro-territórios e ilhas com populações diminutas, que distorciam as médias regionais.
* **Separação de Grandezas:** Estratificação estrita entre as métricas de volume absoluto (`Counts`) e métricas relativas (`Rate per 100,000 population`). Volumes absolutos foram utilizados para somatórios globais e regionais, enquanto taxas relativas foram aplicadas para comparações justas entre países.
* **Tratamento de Dados Sócio-Econômicos (Notebook):** Cruzamento da base de homicídios com dados do Banco Mundial (PIB per capita, Índice Gini e IDH), aplicando transformações logaritmizadas para normalização de escala.

---

## 🔍 Análise Exploratória (EDA)
O módulo descritivo do dashboard responde a 10 questionamentos fundamentais sobre a dinâmica da violência global, extraindo insights como:
* O ranking das nações mais violentas do mundo (Média 2018-2022).
* Mapeamento crítico focado em feminicídio e violência contra a mulher.
* Análise da evolução da taxa de homicídios no Brasil ao longo de uma década.
* Identificação de sub-regiões pacíficas versus zonas de alta letalidade.

---

## 📈 Modelagem de Regressão e Predição (2023-2026)
Utilizando o `scikit-learn`, construímos modelos preditivos baseados no histórico de 10 anos, gerando os seguintes cenários futuros:
* **Regressão Linear e Polinomial (Grau 2):** Aplicadas individualmente por país no Simulador Interativo, permitindo comparar a aderência da linha de tendência versus a curvatura polinomial.
* **Análise por Sexo:** Projeções separadas para vítimas masculinas e femininas, incluindo a modelagem da Razão de Homicídios (Feminino/Masculino).
* **Macrotendências Econômicas e Regionais:** Previsão de volume de letalidade nas 10 maiores economias do mundo e projeções comparativas entre os continentes.

---

## 📊 Data App (Dashboard)
A interface interativa foi construída com **Streamlit** e **Plotly**, focando em usabilidade e design analítico de alta performance:
* **Layout em Abas (Tabs):** Organização lógica das visualizações globais, regionais e individuais sem poluir a interface.
* **Estilização Dinâmica:** Gráficos adaptados com o template `plotly_white`, separação visual clara entre dados reais e previsões (linhas tracejadas), e anotações delimitando o início do período preditivo.

---

## 🚀 Como Executar o Projeto Localmente

O script bash abaixo centraliza todo o fluxo de execução. Ele clona o repositório, cria um ambiente virtual isolado e instala todas as dependências corretamente antes de iniciar a aplicação.

Abra o seu terminal e cole o bloco completo:

```bash
git clone [https://github.com/Felype-byte/Trabalho_01_Analise_Exploratoria_Regressao.git](https://github.com/Felype-byte/Trabalho_01_Analise_Exploratoria_Regressao.git) && \
cd Trabalho_01_Analise_Exploratoria_Regressao && \
python -m venv venv && \
source venv/bin/activate || .\venv\Scripts\activate && \
pip install --upgrade pip && \
pip install streamlit pandas numpy scikit-learn plotly openpyxl seaborn && \
clear && \
echo "🚀 Iniciando o Data App no seu navegador..." && \
streamlit run DataApp.py