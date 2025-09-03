import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Importar dados
df = pd.read_csv("medical_examination.csv")


# 1. Adicionar coluna 'overweight'
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop(columns=['BMI'], inplace=True)


# 2. Normalizar dados: 0 sempre bom, 1 sempre ruim
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 3. Função para o gráfico categórico
def draw_cat_plot():
    # Converter para formato longo
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # Agrupar e contar
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # Desenhar gráfico categórico
    g = sns.catplot(
        x="variable", y="total", hue="value", col="cardio",
        data=df_cat, kind="bar"
    )

    fig = g.fig
    return fig


# 4. Função para o heatmap
def draw_heat_map():
    # Filtrar dados incorretos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Matriz de correlação
    corr = df_heat.corr()

    # Máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configuração da figura
    fig, ax = plt.subplots(figsize=(12, 12))

    # Heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f",
        square=True, center=0, cbar_kws={"shrink": 0.5}
    )

    return fig
