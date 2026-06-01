import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


COLUMNAS_CATEGORICAS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "y",
]


def cargar_datos(ruta_entrada):
    try:
        return pd.read_csv(ruta_entrada) 
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta_entrada}")


def normalizar_categoricas(dataframe: pd.DataFrame) -> pd.DataFrame:
    resultado = dataframe.copy()


    for columna in COLUMNAS_CATEGORICAS:
        if columna in resultado.columns:
            resultado[columna] = resultado[columna].astype(str).str.lower().str.strip()

    if "job" in resultado.columns:
        resultado["job"] = resultado["job"].str.replace("admin.", "administrative", regex=False)

    if "marital" in resultado.columns:
        resultado["marital"] = resultado["marital"].str.replace("div.", "divorced", regex=False)

    if "education" in resultado.columns:
        resultado["education"] = resultado["education"].str.replace("sec.", "secondary", regex=False)
        resultado.loc[resultado["education"] == "unk", "education"] = "unknown"

    if "contact" in resultado.columns:
        resultado.loc[resultado["contact"] == "phone", "contact"] = "telephone"
        resultado.loc[resultado["contact"] == "mobile", "contact"] = "cellular"

    if "poutcome" in resultado.columns:
        resultado.loc[resultado["poutcome"] == "unk", "poutcome"] = "unknown"

    return resultado


def filtrar_registros(dataframe):
    resultado = dataframe.copy()
    reporte = {
        "eliminadas_por_na": 0,
        "eliminadas_por_duplicados": 0,
        "eliminadas_por_filtros": 0,
    }

    antes = len(resultado)
    resultado = resultado.dropna().copy()
    reporte["eliminadas_por_na"] = antes - len(resultado)

    antes = len(resultado)
    resultado = resultado.drop_duplicates().copy()
    reporte["eliminadas_por_duplicados"] = antes - len(resultado)

    condiciones = []
    if "age" in resultado.columns:
        condiciones.append(resultado["age"].between(18, 100))
    if "duration" in resultado.columns:
        condiciones.append(resultado["duration"] > 0)
    if "previous" in resultado.columns:
        condiciones.append(resultado["previous"] <= 100)
    if "pdays" in resultado.columns:
        condiciones.append(resultado["pdays"] >= -1)

    if condiciones:
        mascara = condiciones[0]
        for condicion in condiciones[1:]:
            mascara = mascara & condicion
        antes = len(resultado)
        resultado = resultado.loc[mascara].copy()
        reporte["eliminadas_por_filtros"] = antes - len(resultado)

    return resultado, reporte


def generar_resumen_inicial(dataframe):
    resumen = {
        "filas_iniciales": int(dataframe.shape[0]),
        "columnas_iniciales": int(dataframe.shape[1]),
        "nulos_por_columna": dataframe.isna().sum().to_dict(),
        "estadisticas_numericas": dataframe.describe().round(2).to_dict(),
    }

    conteos_categoricos = {}
    for columna in COLUMNAS_CATEGORICAS:
        if columna in dataframe.columns:
            conteos_categoricos[columna] = dataframe[columna].value_counts(dropna=False).head(10).to_dict()

    resumen["conteos_categoricos"] = conteos_categoricos
    return resumen


def guardar_reporte(ruta_reporte, reporte):
    with open(ruta_reporte, "w", encoding="utf-8") as archivo:
        archivo.write("REPORTE DE LIMPIEZA\n")
        archivo.write("=" * 20 + "\n")
        for clave, valor in reporte.items():
            archivo.write(f"{clave}: {valor}\n")


def crear_graficas_limpieza(reporte_limpieza, filas_iniciales, filas_finales) -> None:
    eliminadas = pd.Series(
        {
            "Nulos": int(reporte_limpieza.get("eliminadas_por_na", 0)),
            "Duplicados": int(reporte_limpieza.get("eliminadas_por_duplicados", 0)),
            "Filtros": int(reporte_limpieza.get("eliminadas_por_filtros", 0)),
        }
    )

    # Gráfica de barras pequeña: filas eliminadas por tipo de limpieza
    plt.figure(figsize=(5.2, 3.2))
    ax_barras = sns.barplot(x=eliminadas.index, y=eliminadas.values, color="#8b5e3c")
    colores = ["#2E86AB", "#D97706", "#C0392B"]
    for barra, color in zip(ax_barras.patches, colores):
        barra.set_color(color)
    plt.title("Registros eliminados por limpieza")
    plt.xlabel("")
    plt.ylabel("Filas")
    for barra in ax_barras.patches:
        valor = int(barra.get_height())
        ax_barras.text(
            barra.get_x() + barra.get_width() / 2,
            valor + max(eliminadas.values) * 0.03,
            f"{valor}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig("Data/Processed/grafica_limpieza_barras.png", dpi=170, bbox_inches="tight")
    plt.close()


def crear_grafica_conversion_segmento(dataframe, columna, ruta_salida, titulo, top_n=None):
    conversion = (
        dataframe.assign(is_yes=(dataframe["y"] == "yes").astype(int))
        .groupby(columna)
        .agg(conversion_rate=("is_yes", "mean"), clients=("is_yes", "size"))
        .reset_index()
    )

    if top_n is not None:
        conversion = conversion.sort_values("conversion_rate", ascending=False).head(top_n)
    else:
        conversion = conversion.sort_values("conversion_rate", ascending=False)

    etiquetas_es = {
        "job": {
            "student": "estudiante",
            "retired": "jubilado",
            "unemployed": "desempleado",
            "management": "gerencia",
            "administrative": "administrativo",
            "self-employed": "autónomo",
            "unknown": "desconocido",
            "technician": "técnico",
        },
        "contact": {
            "cellular": "celular",
            "telephone": "teléfono",
            "unknown": "desconocido",
        },
        "poutcome": {
            "success": "éxito",
            "failure": "fracaso",
            "other": "otro",
            "unknown": "desconocido",
        },
    }

    if columna in etiquetas_es:
        conversion = conversion.copy()
        conversion[columna] = conversion[columna].replace(etiquetas_es[columna])

    plt.figure(figsize=(5.6, 3.6))
    ax = sns.barplot(data=conversion, x="conversion_rate", y=columna, color="#2E86AB")
    plt.title(titulo)
    plt.xlabel("Tasa de conversión")
    plt.ylabel("")
    plt.xlim(0, max(conversion["conversion_rate"]) * 1.15)

    for barra in ax.patches:
        valor = barra.get_width()
        ax.text(valor + 0.005, barra.get_y() + barra.get_height() / 2, f"{valor:.1%}", va="center")

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=170, bbox_inches="tight")
    plt.close()


def main() -> None:
    ruta_entrada = "dataset_banco.csv"
    ruta_salida = "Data/Processed/dataset_banco_limpio.csv"
    ruta_reporte = "Data/Processed/reporte_limpieza.txt"

    try:
        data = cargar_datos(ruta_entrada)
    except FileNotFoundError as error:
        print(error)
        return

    print("Leyendo el dataset...")
    resumen_inicial = generar_resumen_inicial(data)

    print("\n1) Vista general")
    print(f"Filas iniciales: {resumen_inicial['filas_iniciales']}")
    print(f"Columnas iniciales: {resumen_inicial['columnas_iniciales']}")
    print(data.head(5))

    print("\n2) Gráfica de respuesta del cliente")
    data_respuesta = data.copy()
    data_respuesta["y_es"] = data_respuesta["y"].replace({"yes": "sí", "no": "no"})
    plt.figure(figsize=(7, 4))
    ax_respuesta = sns.countplot(x="y_es", data=data_respuesta, color="#2EAB3B")
    plt.title("Clientes interesados y no interesados en adquirir el producto")
    plt.xlabel("Respuesta del cliente")
    plt.ylabel("Cantidad de clientes")
    total_clientes = len(data_respuesta)
    alturas = [barra.get_height() for barra in ax_respuesta.patches]
    minima = min(alturas)
    for barra in ax_respuesta.patches:
        valor = int(barra.get_height())
        porcentaje = valor / total_clientes * 100
        if valor == minima:
            barra.set_color("#C0392B")
        ax_respuesta.text(
            barra.get_x() + barra.get_width() / 2,
            valor / 2,
            f"{porcentaje:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig("Data/Processed/grafica_respuesta_clientes.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("\n3) Limpieza de datos")
    data_limpia = normalizar_categoricas(data)
    data_limpia, reporte_limpieza = filtrar_registros(data_limpia)

    print(f"Filas eliminadas por nulos: {reporte_limpieza['eliminadas_por_na']}")
    print(f"Filas eliminadas por duplicados: {reporte_limpieza['eliminadas_por_duplicados']}")
    print(f"Filas eliminadas por filtros simples: {reporte_limpieza['eliminadas_por_filtros']}")

    crear_graficas_limpieza(reporte_limpieza, resumen_inicial["filas_iniciales"], len(data_limpia))

    print("\n4) Gráfica de conversión")
    conversion_por_trabajo = (
        data_limpia.assign(is_yes=(data_limpia["y"] == "yes").astype(int))
        .groupby("job")
        .agg(conversion_rate=("is_yes", "mean"), clients=("is_yes", "size"))
        .reset_index()
        .query("clients >= 100")
        .sort_values("conversion_rate", ascending=False)
        .head(8)
    )

    print("\nTabla de conversión por perfil laboral")
    print(conversion_por_trabajo.to_string(index=False))

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=conversion_por_trabajo, x="job", y="conversion_rate", color="#2E86AB")
    plt.title("Tasa de conversión por perfil laboral")
    plt.xlabel("Perfil laboral")
    plt.ylabel("Tasa de conversión")
    plt.ylim(0, max(conversion_por_trabajo["conversion_rate"]) * 1.15)

    for barra in ax.patches:
        valor = barra.get_height()
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.005, f"{valor:.1%}", ha="center")

    plt.tight_layout()
    plt.savefig("Data/Processed/grafica_comparativa_conversion.png", dpi=160, bbox_inches="tight")
    plt.close()

    crear_grafica_conversion_segmento(
        data_limpia,
        "job",
        "Data/Processed/grafica_conversion_job.png",
        "Conversión por perfil laboral",
        top_n=3,
    )
    crear_grafica_conversion_segmento(
        data_limpia,
        "contact",
        "Data/Processed/grafica_conversion_contacto.png",
        "Conversión por canal de contacto",
    )
    crear_grafica_conversion_segmento(
        data_limpia,
        "poutcome",
        "Data/Processed/grafica_conversion_poutcome.png",
        "Conversión según campaña previa",
    )
    

    data_limpia.to_csv(ruta_salida, index=False)

    reporte_final = {
        **resumen_inicial,
        **reporte_limpieza,
        "filas_finales": int(data_limpia.shape[0]),
        "columnas_finales": int(data_limpia.shape[1]),
        "ruta_salida": str(ruta_salida),
    }
    guardar_reporte(ruta_reporte, reporte_final)

    print("\n5) Resultado final")
    print(f"Dataset limpio guardado en: {ruta_salida}")
    print(f"Reporte guardado en: {ruta_reporte}")
    print(f"Tamaño final: {data_limpia.shape}")


if __name__ == "__main__":
    main()
