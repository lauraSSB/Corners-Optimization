from CornerAnalyzer import CornerAnalyzer


def main():
    """Función principal para ejecutar el análisis de corners"""

    print("ANÁLISIS DE TIROS DE ESQUINA - STATSBOMB")

    analyzer = CornerAnalyzer()

    print("\n--- PASO 1: Obteniendo competiciones disponibles ---")
    comps = analyzer.get_competitions()
    print(comps)

    competition_id = int(input("\nIngresa el competition_id: "))
    season_id = int(input("Ingresa el season_id: "))

    corners = analyzer.extract_corners_from_season(
        competition_id=competition_id,
        season_id=season_id
    )



    if corners.empty:
        print("\nNo se encontraron corners. Verifica los IDs de competición y temporada.")
        return

    corners.to_csv('corners_analysis_excel.csv', index=False)

    print(f"\n✓ DataFrame creado con {len(corners)} corners")

if __name__ == "__main__":
    main()