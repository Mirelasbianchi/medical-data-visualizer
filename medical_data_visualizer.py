# ============================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================
import pandas as pd              # Manipulação e análise de dados em DataFrames
import seaborn as sns            # Criação de gráficos estatísticos
import matplotlib.pyplot as plt  # Ferramentas gráficas
import numpy as np               # Operações matemáticas e arrays


# ============================================
# IMPORTAÇÃO DOS DADOS
# ============================================
# Carrega os dados médicos a partir de um arquivo CSV
df = pd.read_csv("medical_examination.csv")


# ============================================
# 1. CRIAR A COLUNA "OVERWEIGHT" (SOBREPESO)
# ============================================
# Fórmula do IMC (Índice de Massa Corporal):
#   BMI = peso / (altura^2)
# Obs: a altura está em centímetros, então convertemos para metros (/100).
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

# Cria uma nova coluna binária:
#   - 0 = peso normal (BMI <= 25)
#   - 1 = sobrepeso (BMI > 25)
df['overweight'] = (df['BMI'] > 25).astype(int)

# Removemos a coluna temporária 'BMI', pois não é mais necessária
df.drop(columns=['BMI'], inplace=True)


# ============================================
# 2. NORMALIZAÇÃO DE DADOS
# ============================================
# As colunas 'cholesterol' e 'gluc' tinham valores:
#   1 = normal
#   2 = acima do normal
#   3 = muito acima do normal
#
# Transformamos em binário:
#   0 = saudável (valor original = 1)
#   1 = não saudável (valor original = 2 ou 3)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# ============================================
# 3. FUNÇÃO PARA GRÁFICO CATEGÓRICO
# ============================================
def draw_cat_plot():
    """
    Cria um gráfico categórico que mostra, para cada fator de risco
    (colesterol, glicose, fumar, álcool, atividade física e sobrepeso),
    a quantidade de pessoas com e sem doenças cardíacas.
    """

    # Transformar os dados no formato "longo" (long-form) usando melt:
    # Cada linha terá:
    #   - cardio (0 ou 1: sem/doença cardíaca)
    #   - variável (ex: cholesterol, gluc, etc.)
    #   - valor (0 = saudável, 1 = não saudável)
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],   # Mantém a coluna "cardio"
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # Agrupar por (cardio, variável, valor) e contar quantos registros existem
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # Criar gráfico de barras categórico
    g = sns.catplot(
        x="variable", y="total", hue="value", col="cardio",
        data=df_cat, kind="bar"
    )

    # Retornar a figura para exibição
    fig = g.fig
    return fig


# ============================================
# 4. FUNÇÃO PARA HEATMAP (MATRIZ DE CORRELAÇÃO)
# ============================================
def draw_heat_map():
    """
    Cria um heatmap (mapa de calor) mostrando a correlação entre
    variáveis médicas, após remover dados inconsistentes e outliers.
    """

    # 1. Filtrar dados inválidos ou extremos:
    # - Pressão diastólica (ap_lo) não pode ser maior que a sistólica (ap_hi).
    # - Altura e peso devem estar entre o intervalo de 2.5% e 97.5%
    #   (remoção de outliers extremos).
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 2. Calcular matriz de correlação
    corr = df_heat.corr()

    # 3. Criar máscara para esconder o triângulo superior da matriz
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 4. Configuração da figura
    fig, ax = plt.subplots(figsize=(12, 12))

    # 5. Criar heatmap com anotações numéricas
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f",
        square=True, center=0, cbar_kws={"shrink": 0.5}
    )

    return fig
