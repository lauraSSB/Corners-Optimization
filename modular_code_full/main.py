from CornerAnalyzer import CornerAnalyzer


def main():
    """Main function to execute corner analysis"""

    print("CORNER KICK ANALYSIS - STATSBOMB")

    analyzer = CornerAnalyzer()

    print("\n--- STEP 1: Getting available competitions ---")
    comps = analyzer.get_competitions()
    print(comps)

    competition_id = int(input("\nEnter competition_id: "))
    season_id = int(input("Enter season_id: "))

    corners = analyzer.extract_corners_from_season(
        competition_id=competition_id,
        season_id=season_id
    )

    if corners.empty:
        print("\nNo corners found. Check competition and season IDs.")
        return

    print(f"\n✓ DataFrame created with {len(corners)} corners")

    # Show summary statistics
    analyzer.get_summary_stats()
  
    # Show zone distribution analysis
    analyzer.analyze_zone_distribution()
  
    # Save to CSV
    analyzer.save_to_csv()
  
    # Show corner distribution chart
    analyzer.plot_corner_distribution()

if __name__ == "__main__":
    main()