"""Marka/desen/mevsim sözlükleri. Kolayca genişletilebilir."""
from __future__ import annotations

# marka adı -> o markaya ait bilinen desen (pattern) adları
BRAND_PATTERNS: dict[str, list[str]] = {
    "Continental": [
        "EcoContact 6", "EcoContact 5", "SportContact 7", "SportContact 6",
        "SportContact 5", "PremiumContact 6", "PremiumContact 5", "AllSeasonContact 2",
        "AllSeasonContact", "WinterContact TS 870", "WinterContact TS 830",
        "UltraContact", "CrossContact ATR", "CrossContact LX", "VanContact",
    ],
    "Goodyear": [
        "EfficientGrip Performance 2", "EfficientGrip Performance", "EfficientGrip",
        "UltraGrip Performance", "UltraGrip 9", "UltraGrip", "Vector 4Seasons Gen-3",
        "Vector 4Seasons", "EagleF1 Asymmetric 6", "EagleF1 Asymmetric 5", "EagleF1",
        "Cargo Vector 2",
    ],
    "Lassa": [
        "Multiways 2", "Multiways", "Driveways 2", "Driveways", "Competus H/L 2",
        "Competus", "Snoways 4", "Snoways 3", "Snoways", "Greenways",
    ],
    "Petlas": [
        "Elegant PT311", "Elegant", "Explero RG02", "Explero", "Progreen PT525",
        "Progreen", "Snowmaster W651", "Snowmaster", "Imperius PT535", "Imperius",
        "Velox Sport PT741",
    ],
    "Michelin": [
        "Primacy 4+", "Primacy 4", "Primacy 3", "Pilot Sport 5", "Pilot Sport 4",
        "CrossClimate 2", "CrossClimate+", "Energy Saver+", "Alpin 6", "Alpin 5",
        "e.Primacy",
    ],
    "Bridgestone": [
        "Turanza 6", "Turanza T005", "Potenza Sport", "Potenza RE050", "Blizzak LM005",
        "Ecopia EP150", "Ecopia EP300", "Weather Control A005", "Duravis",
    ],
    "Pirelli": [
        "Cinturato P7", "Cinturato P1", "P Zero", "Scorpion Verde", "Scorpion Winter",
        "Winter Sottozero 3", "Powergy",
    ],
    "Hankook": [
        "Ventus Prime 4", "Ventus S1 evo3", "Kinergy Eco2", "Kinergy 4S 2",
        "Winter i*cept RS3", "Winter i*cept evo3", "Dynapro HP2",
    ],
    "Kumho": [
        "Ecsta HS51", "Ecsta PS71", "Solus TA31", "Solus TA21", "Solus 4S",
        "Crugen HP71", "EcoWing ES31", "EcoWing ES01",
    ],
    "Falken": ["Ziex ZE310", "Ziex ZE914", "Wildpeak A/T", "Euroall Season AS210"],
    "Nokian": ["Hakkapeliitta R5", "Hakkapeliitta 10", "Weatherproof", "Nordman"],
    "Sava": ["Eskimo S3+", "Perfecta", "Intensa UHP2", "Intensa HP2", "Trenta"],
    "Debica": ["Presto UHP2", "Presto HP2", "Passio 2", "Frigo 2"],
    "Barum": ["Bravuris 5HM", "Bravuris 4x4", "Polaris 5", "Quartaris 5"],
    "Fulda": ["EcoControl HP2", "Kristall Control HP3", "MultiControl", "Conveo Tour 2"],
    "Uniroyal": ["RainSport 5", "AllSeasonExpert 2", "WinterExpert"],
    "Semperit": ["Speed-Life 3", "Comfort-Life 2", "Master-Grip 2", "Van-Life 3"],
    "Vredestein": ["Ultrac Vorti", "Quatrac", "Wintrac Pro", "Sportrac 5"],
    "Kormoran": ["Road Performance", "SUV Summer", "All Season"],
    "Gislaved": ["Speed 626", "EuroFrost 6", "Soft*Frost 200"],
    "Matador": ["MP47 Hectorra 3", "MP62 All Weather 2", "MP93 Nordicca"],
    "Firestone": ["Roadhawk 2", "Multihawk 2", "Winterhawk 4"],
    "General Tire": ["Altimax One", "Altimax Winter 3", "Grabber AT3"],
    "BFGoodrich": ["Advantage", "g-Force Winter2", "All-Terrain T/A KO2"],
    "Toyo": ["Proxes Comfort", "Proxes Sport", "Observe S944", "Open Country A/T3"],
    "Yokohama": ["BluEarth-GT AE51", "Advan Sport V107", "iceGUARD iG65"],
    "Nexen": ["N'Fera Primus", "N'Blue HD Plus", "Winguard Sport 2"],
    "Nankang": ["AS-1", "AS-2+", "Winter Activa SV-3"],
    "Maxxis": ["Premitra 5", "Premitra HP5", "Premitra Ice 5"],
    "Sailun": ["Atrezzo ZSR", "Atrezzo Elite", "WinterProof WSL-2"],
    "Starmaxx": ["Incurro A/S ST450", "Novaro ST532", "Ultrasport ST810"],
    "Rotalla": ["Setula S-Race", "Setula W-Race"],
    "Linglong": ["Green-Max", "Winter Grip"],
    "Kenda": ["Vezda UHP", "Wintergen 2"],
    "Dunlop": ["Sport BluResponse", "SP Winter Sport", "Grandtrek"],
    "Waterfall": ["Eco Dynamic", "Quattro", "Sport Xtreme"],
    "Crosswind": ["Comfort Peak", "Endura Touring", "All Ranger", "HP010"],
}

BRANDS = list(BRAND_PATTERNS.keys())

ALL_PATTERNS: list[tuple[str, str]] = [
    (pattern, brand) for brand, patterns in BRAND_PATTERNS.items() for pattern in patterns
]

# desen adında geçen anahtar kelime -> mevsim
SEASON_KEYWORDS: list[tuple[str, str]] = [
    ("allseason", "4 Mevsim"),
    ("all season", "4 Mevsim"),
    ("all-season", "4 Mevsim"),
    ("4season", "4 Mevsim"),
    ("4 season", "4 Mevsim"),
    ("multiways", "4 Mevsim"),
    ("crossclimate", "4 Mevsim"),
    ("weatherproof", "4 Mevsim"),
    ("vector 4seasons", "4 Mevsim"),
    ("quatrac", "4 Mevsim"),
    ("all weather", "4 Mevsim"),
    ("winter", "Kış"),
    ("snow", "Kış"),
    ("blizzak", "Kış"),
    ("hakkapeliitta", "Kış"),
    ("ultragrip", "Kış"),
    ("icept", "Kış"),
    ("i-cept", "Kış"),
    ("frigo", "Kış"),
    ("nordicca", "Kış"),
    ("wintrac", "Kış"),
    ("nordman", "Kış"),
    ("eco", "Yaz"),
    ("sport", "Yaz"),
    ("premium", "Yaz"),
    ("primacy", "Yaz"),
    ("turanza", "Yaz"),
    ("cinturato", "Yaz"),
    ("efficientgrip performance", "Yaz"),
    ("pilot sport", "Yaz"),
    ("potenza", "Yaz"),
    ("eagle f1", "Yaz"),
    ("p zero", "Yaz"),
    ("ventus", "Yaz"),
    ("ecsta", "Yaz"),
    ("proxes", "Yaz"),
    ("elegant", "Yaz"),
    ("explero", "Yaz"),
]

# metinde doğrudan geçebilecek mevsim işaretleri
SEASON_TEXT_MARKERS: list[tuple[str, str]] = [
    ("3PMSF", "Kış"),
    ("M+S", "4 Mevsim"),
    ("M&S", "4 Mevsim"),
    ("ALL SEASON", "4 Mevsim"),
    ("4 SEASON", "4 Mevsim"),
    ("ALL WEATHER", "4 Mevsim"),
]
