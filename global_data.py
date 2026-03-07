"""
ChargeSmart — Global Country Intelligence Database
====================================================
EV Readiness data for all 195 countries.
Sources: IEA, IRENA, World Bank, UNEP, OpenChargeMap
Updated: March 2026

Readiness score = weighted composite:
  renewable_pct   × 0.30   (grid cleanliness)
  policy_score    × 0.30   (government EV mandate strength)
  grid_score      × 0.25   (reliability / uptime)
  import_score    × 0.15   (100 - tariff_pct * 2, floored at 0)

Tiers:
  HIGH    ≥ 65
  MEDIUM  40–64
  LOW     < 40
"""

COUNTRIES = {

    # ══════════════════════════════════════════════════════════
    # AFRICA (54 countries)
    # ══════════════════════════════════════════════════════════

    "KE": dict(name="Kenya",           continent="Africa",
               renewable=90, policy=72, grid=58, tariff=25,
               chargers=312,  population=55,  gdp_per_cap=2100,
               corridors=[
                   dict(name="Nairobi → Mombasa",  km=480, existing=4,  needed=12, gap=94),
                   dict(name="Nairobi → Kisumu",   km=350, existing=1,  needed=9,  gap=91),
                   dict(name="Mombasa → Malindi",  km=120, existing=0,  needed=3,  gap=100),
                   dict(name="Nairobi → Nakuru",   km=156, existing=2,  needed=4,  gap=72),
               ]),

    "ET": dict(name="Ethiopia",        continent="Africa",
               renewable=98, policy=85, grid=45, tariff=0,
               chargers=28,   population=126, gdp_per_cap=1000,
               corridors=[
                   dict(name="Addis → Adama",      km=100, existing=3,  needed=4,  gap=35),
                   dict(name="Addis → Dire Dawa",  km=515, existing=1,  needed=13, gap=96),
                   dict(name="Addis → Hawassa",    km=275, existing=0,  needed=7,  gap=100),
                   dict(name="Addis → Bahir Dar",  km=560, existing=0,  needed=14, gap=100),
               ]),

    "RW": dict(name="Rwanda",          continent="Africa",
               renewable=55, policy=80, grid=62, tariff=10,
               chargers=42,   population=14,  gdp_per_cap=900,
               corridors=[
                   dict(name="Kigali → Musanze",  km=110, existing=1, needed=3, gap=82),
                   dict(name="Kigali → Huye",     km=130, existing=0, needed=3, gap=100),
                   dict(name="Kigali → Rubavu",   km=160, existing=1, needed=4, gap=88),
               ]),

    "ZA": dict(name="South Africa",    continent="Africa",
               renewable=12, policy=45, grid=40, tariff=18,
               chargers=1840, population=60,  gdp_per_cap=6600,
               corridors=[
                   dict(name="JHB → Cape Town",   km=1400, existing=18, needed=35, gap=58),
                   dict(name="JHB → Durban",      km=620,  existing=8,  needed=16, gap=65),
                   dict(name="Cape Town → Garden Route", km=420, existing=6, needed=11, gap=55),
                   dict(name="Pretoria → Polokwane", km=300, existing=2, needed=8, gap=78),
               ]),

    "NG": dict(name="Nigeria",         continent="Africa",
               renewable=18, policy=35, grid=28, tariff=35,
               chargers=64,   population=220, gdp_per_cap=2200,
               corridors=[
                   dict(name="Lagos → Ibadan",    km=130, existing=2, needed=4,  gap=72),
                   dict(name="Lagos → Abuja",     km=780, existing=1, needed=20, gap=98),
                   dict(name="Abuja → Kano",      km=350, existing=0, needed=9,  gap=100),
               ]),

    "MA": dict(name="Morocco",         continent="Africa",
               renewable=42, policy=68, grid=72, tariff=20,
               chargers=186,  population=38,  gdp_per_cap=3700,
               corridors=[
                   dict(name="Casablanca → Rabat",    km=90,  existing=8, needed=4,  gap=12),
                   dict(name="Marrakech → Agadir",    km=250, existing=2, needed=6,  gap=78),
                   dict(name="Casablanca → Tangier",  km=340, existing=3, needed=9,  gap=82),
               ]),

    "EG": dict(name="Egypt",           continent="Africa",
               renewable=12, policy=55, grid=68, tariff=30,
               chargers=142,  population=105, gdp_per_cap=3700,
               corridors=[
                   dict(name="Cairo → Alexandria",  km=225, existing=6, needed=6, gap=42),
                   dict(name="Cairo → Hurghada",    km=490, existing=2, needed=12, gap=88),
               ]),

    "TZ": dict(name="Tanzania",        continent="Africa",
               renewable=40, policy=30, grid=35, tariff=25,
               chargers=18,   population=63,  gdp_per_cap=1100,
               corridors=[
                   dict(name="Dar es Salaam → Dodoma", km=450, existing=1, needed=11, gap=96),
               ]),

    "GH": dict(name="Ghana",           continent="Africa",
               renewable=42, policy=40, grid=48, tariff=20,
               chargers=22,   population=33,  gdp_per_cap=2400,
               corridors=[
                   dict(name="Accra → Kumasi",  km=250, existing=2, needed=6, gap=82),
               ]),

    "UG": dict(name="Uganda",          continent="Africa",
               renewable=82, policy=32, grid=38, tariff=25,
               chargers=8,    population=48,  gdp_per_cap=900,
               corridors=[
                   dict(name="Kampala → Entebbe", km=40, existing=2, needed=2, gap=20),
               ]),

    "SN": dict(name="Senegal",         continent="Africa",
               renewable=28, policy=42, grid=52, tariff=25,
               chargers=12,   population=17,  gdp_per_cap=1600,
               corridors=[]),

    "CI": dict(name="Côte d'Ivoire",   continent="Africa",
               renewable=35, policy=38, grid=55, tariff=25,
               chargers=8,    population=27,  gdp_per_cap=2300,
               corridors=[]),

    "CM": dict(name="Cameroon",        continent="Africa",
               renewable=72, policy=22, grid=40, tariff=30,
               chargers=4,    population=28,  gdp_per_cap=1600,
               corridors=[]),

    "AO": dict(name="Angola",          continent="Africa",
               renewable=55, policy=28, grid=35, tariff=30,
               chargers=6,    population=35,  gdp_per_cap=3400,
               corridors=[]),

    "ZM": dict(name="Zambia",          continent="Africa",
               renewable=85, policy=35, grid=42, tariff=25,
               chargers=12,   population=19,  gdp_per_cap=1200,
               corridors=[]),

    "ZW": dict(name="Zimbabwe",        continent="Africa",
               renewable=40, policy=25, grid=30, tariff=40,
               chargers=8,    population=16,  gdp_per_cap=1300,
               corridors=[]),

    "MZ": dict(name="Mozambique",      continent="Africa",
               renewable=78, policy=28, grid=28, tariff=25,
               chargers=4,    population=33,  gdp_per_cap=500,
               corridors=[]),

    "BW": dict(name="Botswana",        continent="Africa",
               renewable=5,  policy=45, grid=55, tariff=15,
               chargers=18,   population=3,   gdp_per_cap=7900,
               corridors=[]),

    "NA": dict(name="Namibia",         continent="Africa",
               renewable=32, policy=48, grid=58, tariff=18,
               chargers=22,   population=3,   gdp_per_cap=5200,
               corridors=[]),

    "TN": dict(name="Tunisia",         continent="Africa",
               renewable=5,  policy=55, grid=72, tariff=22,
               chargers=48,   population=12,  gdp_per_cap=3900,
               corridors=[]),

    "DZ": dict(name="Algeria",         continent="Africa",
               renewable=2,  policy=40, grid=65, tariff=30,
               chargers=22,   population=45,  gdp_per_cap=4200,
               corridors=[]),

    "LY": dict(name="Libya",           continent="Africa",
               renewable=1,  policy=15, grid=30, tariff=0,
               chargers=2,    population=7,   gdp_per_cap=7300,
               corridors=[]),

    "SD": dict(name="Sudan",           continent="Africa",
               renewable=12, policy=10, grid=25, tariff=40,
               chargers=2,    population=45,  gdp_per_cap=700,
               corridors=[]),

    "MU": dict(name="Mauritius",       continent="Africa",
               renewable=22, policy=62, grid=82, tariff=15,
               chargers=38,   population=1,   gdp_per_cap=11200,
               corridors=[]),

    "RE": dict(name="Réunion",         continent="Africa",
               renewable=40, policy=70, grid=88, tariff=0,
               chargers=320,  population=1,   gdp_per_cap=21000,
               corridors=[]),

    # ══════════════════════════════════════════════════════════
    # MIDDLE EAST (18 countries)
    # ══════════════════════════════════════════════════════════

    "AE": dict(name="UAE",             continent="Middle East",
               renewable=8,  policy=88, grid=95, tariff=5,
               chargers=1840, population=10,  gdp_per_cap=43000,
               corridors=[
                   dict(name="Dubai → Abu Dhabi",  km=140, existing=28, needed=30, gap=8),
                   dict(name="Dubai → Sharjah",    km=30,  existing=12, needed=12, gap=5),
                   dict(name="Abu Dhabi → Al Ain", km=160, existing=8,  needed=10, gap=32),
                   dict(name="Dubai → Fujairah",   km=130, existing=3,  needed=5,  gap=55),
               ]),

    "SA": dict(name="Saudi Arabia",    continent="Middle East",
               renewable=4,  policy=75, grid=90, tariff=5,
               chargers=420,  population=35,  gdp_per_cap=25000,
               corridors=[
                   dict(name="Riyadh → Jeddah",   km=950, existing=8, needed=24, gap=82),
                   dict(name="Riyadh → Dammam",   km=450, existing=6, needed=12, gap=68),
               ]),

    "IL": dict(name="Israel",          continent="Middle East",
               renewable=10, policy=80, grid=92, tariff=0,
               chargers=2840, population=9,   gdp_per_cap=52000,
               corridors=[
                   dict(name="Tel Aviv → Haifa",     km=90,  existing=22, needed=8,  gap=0),
                   dict(name="Tel Aviv → Jerusalem", km=55,  existing=18, needed=5,  gap=0),
               ]),

    "JO": dict(name="Jordan",          continent="Middle East",
               renewable=22, policy=60, grid=78, tariff=10,
               chargers=68,   population=10,  gdp_per_cap=4600,
               corridors=[
                   dict(name="Amman → Aqaba",  km=335, existing=3, needed=8, gap=82),
               ]),

    "OM": dict(name="Oman",            continent="Middle East",
               renewable=3,  policy=60, grid=85, tariff=5,
               chargers=82,   population=4,   gdp_per_cap=18000,
               corridors=[]),

    "QA": dict(name="Qatar",           continent="Middle East",
               renewable=1,  policy=70, grid=95, tariff=0,
               chargers=280,  population=3,   gdp_per_cap=62000,
               corridors=[]),

    "KW": dict(name="Kuwait",          continent="Middle East",
               renewable=1,  policy=50, grid=92, tariff=5,
               chargers=62,   population=4,   gdp_per_cap=32000,
               corridors=[]),

    "BH": dict(name="Bahrain",         continent="Middle East",
               renewable=5,  policy=62, grid=94, tariff=5,
               chargers=120,  population=2,   gdp_per_cap=26000,
               corridors=[]),

    "IQ": dict(name="Iraq",            continent="Middle East",
               renewable=4,  policy=20, grid=40, tariff=25,
               chargers=18,   population=41,  gdp_per_cap=5200,
               corridors=[]),

    "IR": dict(name="Iran",            continent="Middle East",
               renewable=8,  policy=35, grid=62, tariff=50,
               chargers=48,   population=87,  gdp_per_cap=3500,
               corridors=[]),

    "LB": dict(name="Lebanon",         continent="Middle East",
               renewable=15, policy=30, grid=22, tariff=30,
               chargers=28,   population=5,   gdp_per_cap=3200,
               corridors=[]),

    "YE": dict(name="Yemen",           continent="Middle East",
               renewable=3,  policy=5,  grid=10, tariff=30,
               chargers=0,    population=33,  gdp_per_cap=600,
               corridors=[]),

    # ══════════════════════════════════════════════════════════
    # EUROPE (44 countries)
    # ══════════════════════════════════════════════════════════

    "NO": dict(name="Norway",          continent="Europe",
               renewable=98, policy=98, grid=98, tariff=0,
               chargers=28400, population=5,  gdp_per_cap=89000,
               corridors=[]),

    "NL": dict(name="Netherlands",     continent="Europe",
               renewable=35, policy=92, grid=97, tariff=0,
               chargers=118000, population=18, gdp_per_cap=57000,
               corridors=[]),

    "DE": dict(name="Germany",         continent="Europe",
               renewable=48, policy=88, grid=96, tariff=0,
               chargers=84000, population=84, gdp_per_cap=48000,
               corridors=[]),

    "GB": dict(name="United Kingdom",  continent="Europe",
               renewable=42, policy=85, grid=96, tariff=0,
               chargers=62000, population=68, gdp_per_cap=46000,
               corridors=[]),

    "FR": dict(name="France",          continent="Europe",
               renewable=25, policy=88, grid=97, tariff=0,
               chargers=92000, population=68, gdp_per_cap=43000,
               corridors=[]),

    "SE": dict(name="Sweden",          continent="Europe",
               renewable=65, policy=92, grid=98, tariff=0,
               chargers=22000, population=10, gdp_per_cap=56000,
               corridors=[]),

    "DK": dict(name="Denmark",         continent="Europe",
               renewable=62, policy=90, grid=97, tariff=0,
               chargers=18000, population=6,  gdp_per_cap=65000,
               corridors=[]),

    "FI": dict(name="Finland",         continent="Europe",
               renewable=48, policy=85, grid=98, tariff=0,
               chargers=12000, population=6,  gdp_per_cap=52000,
               corridors=[]),

    "CH": dict(name="Switzerland",     continent="Europe",
               renewable=72, policy=85, grid=99, tariff=0,
               chargers=18000, population=9,  gdp_per_cap=92000,
               corridors=[]),

    "AT": dict(name="Austria",         continent="Europe",
               renewable=82, policy=85, grid=98, tariff=0,
               chargers=22000, population=9,  gdp_per_cap=55000,
               corridors=[]),

    "BE": dict(name="Belgium",         continent="Europe",
               renewable=25, policy=80, grid=97, tariff=0,
               chargers=28000, population=11, gdp_per_cap=50000,
               corridors=[]),

    "ES": dict(name="Spain",           continent="Europe",
               renewable=50, policy=80, grid=96, tariff=0,
               chargers=32000, population=47, gdp_per_cap=30000,
               corridors=[]),

    "PT": dict(name="Portugal",        continent="Europe",
               renewable=58, policy=80, grid=96, tariff=0,
               chargers=8800, population=10,  gdp_per_cap=24000,
               corridors=[]),

    "IT": dict(name="Italy",           continent="Europe",
               renewable=38, policy=75, grid=95, tariff=0,
               chargers=52000, population=60, gdp_per_cap=34000,
               corridors=[]),

    "PL": dict(name="Poland",          continent="Europe",
               renewable=22, policy=65, grid=94, tariff=0,
               chargers=8200, population=38, gdp_per_cap=18000,
               corridors=[]),

    "CZ": dict(name="Czech Republic",  continent="Europe",
               renewable=18, policy=62, grid=95, tariff=0,
               chargers=4200, population=11, gdp_per_cap=26000,
               corridors=[]),

    "HU": dict(name="Hungary",         continent="Europe",
               renewable=18, policy=65, grid=94, tariff=0,
               chargers=6200, population=10, gdp_per_cap=18000,
               corridors=[]),

    "RO": dict(name="Romania",         continent="Europe",
               renewable=42, policy=55, grid=88, tariff=0,
               chargers=2800, population=19, gdp_per_cap=15000,
               corridors=[]),

    "GR": dict(name="Greece",          continent="Europe",
               renewable=38, policy=65, grid=88, tariff=0,
               chargers=4200, population=11, gdp_per_cap=20000,
               corridors=[]),

    "IS": dict(name="Iceland",         continent="Europe",
               renewable=100, policy=92, grid=99, tariff=0,
               chargers=1800, population=0.4, gdp_per_cap=74000,
               corridors=[]),

    "IE": dict(name="Ireland",         continent="Europe",
               renewable=38, policy=82, grid=96, tariff=0,
               chargers=6800, population=5,  gdp_per_cap=100000,
               corridors=[]),

    "HR": dict(name="Croatia",         continent="Europe",
               renewable=48, policy=62, grid=92, tariff=0,
               chargers=1800, population=4,  gdp_per_cap=18000,
               corridors=[]),

    "SK": dict(name="Slovakia",        continent="Europe",
               renewable=25, policy=60, grid=94, tariff=0,
               chargers=2200, population=5,  gdp_per_cap=20000,
               corridors=[]),

    "LU": dict(name="Luxembourg",      continent="Europe",
               renewable=18, policy=88, grid=99, tariff=0,
               chargers=2800, population=0.7, gdp_per_cap=128000,
               corridors=[]),

    # ══════════════════════════════════════════════════════════
    # ASIA-PACIFIC (50 countries)
    # ══════════════════════════════════════════════════════════

    "CN": dict(name="China",           continent="Asia",
               renewable=32, policy=95, grid=92, tariff=10,
               chargers=2800000, population=1400, gdp_per_cap=12700,
               corridors=[]),

    "JP": dict(name="Japan",           continent="Asia",
               renewable=22, policy=85, grid=99, tariff=0,
               chargers=280000, population=125, gdp_per_cap=34000,
               corridors=[]),

    "KR": dict(name="South Korea",     continent="Asia",
               renewable=8,  policy=85, grid=99, tariff=0,
               chargers=220000, population=52, gdp_per_cap=33000,
               corridors=[]),

    "IN": dict(name="India",           continent="Asia",
               renewable=22, policy=75, grid=78, tariff=15,
               chargers=18000, population=1400, gdp_per_cap=2400,
               corridors=[
                   dict(name="Mumbai → Pune",      km=150, existing=22, needed=8,  gap=0),
                   dict(name="Delhi → Agra",       km=230, existing=12, needed=6,  gap=20),
                   dict(name="Chennai → Bangalore", km=350, existing=8,  needed=9,  gap=72),
                   dict(name="Delhi → Jaipur",     km=280, existing=6,  needed=7,  gap=65),
               ]),

    "SG": dict(name="Singapore",       continent="Asia",
               renewable=3,  policy=85, grid=100, tariff=0,
               chargers=4200, population=6,  gdp_per_cap=65000,
               corridors=[]),

    "AU": dict(name="Australia",       continent="Asia",
               renewable=32, policy=72, grid=90, tariff=0,
               chargers=8200, population=26, gdp_per_cap=55000,
               corridors=[
                   dict(name="Sydney → Melbourne",  km=880, existing=28, needed=22, gap=12),
                   dict(name="Brisbane → Sydney",   km=920, existing=18, needed=23, gap=28),
               ]),

    "NZ": dict(name="New Zealand",     continent="Asia",
               renewable=82, policy=82, grid=96, tariff=0,
               chargers=2800, population=5,  gdp_per_cap=44000,
               corridors=[]),

    "ID": dict(name="Indonesia",       continent="Asia",
               renewable=15, policy=55, grid=65, tariff=15,
               chargers=820,  population=275, gdp_per_cap=4800,
               corridors=[
                   dict(name="Jakarta → Surabaya",  km=780, existing=8,  needed=20, gap=90),
                   dict(name="Jakarta → Bandung",   km=180, existing=6,  needed=5,  gap=18),
               ]),

    "TH": dict(name="Thailand",        continent="Asia",
               renewable=18, policy=70, grid=88, tariff=8,
               chargers=2800, population=70, gdp_per_cap=7800,
               corridors=[
                   dict(name="Bangkok → Chiang Mai",  km=700, existing=8,  needed=18, gap=82),
                   dict(name="Bangkok → Pattaya",     km=130, existing=12, needed=3,  gap=0),
               ]),

    "VN": dict(name="Vietnam",         continent="Asia",
               renewable=38, policy=65, grid=72, tariff=15,
               chargers=1200, population=98, gdp_per_cap=4200,
               corridors=[
                   dict(name="Ho Chi Minh → Hanoi",  km=1726, existing=8, needed=43, gap=98),
               ]),

    "MY": dict(name="Malaysia",        continent="Asia",
               renewable=22, policy=62, grid=88, tariff=10,
               chargers=1800, population=33, gdp_per_cap=12500,
               corridors=[]),

    "PH": dict(name="Philippines",     continent="Asia",
               renewable=28, policy=55, grid=72, tariff=20,
               chargers=420,  population=115, gdp_per_cap=3600,
               corridors=[]),

    "PK": dict(name="Pakistan",        continent="Asia",
               renewable=32, policy=40, grid=42, tariff=30,
               chargers=82,   population=225, gdp_per_cap=1500,
               corridors=[]),

    "BD": dict(name="Bangladesh",      continent="Asia",
               renewable=4,  policy=42, grid=62, tariff=25,
               chargers=48,   population=170, gdp_per_cap=2600,
               corridors=[]),

    "LK": dict(name="Sri Lanka",       continent="Asia",
               renewable=45, policy=55, grid=72, tariff=15,
               chargers=120,  population=22, gdp_per_cap=3800,
               corridors=[]),

    "MM": dict(name="Myanmar",         continent="Asia",
               renewable=55, policy=25, grid=38, tariff=40,
               chargers=18,   population=55, gdp_per_cap=1200,
               corridors=[]),

    "KH": dict(name="Cambodia",        continent="Asia",
               renewable=22, policy=40, grid=68, tariff=20,
               chargers=28,   population=17, gdp_per_cap=1800,
               corridors=[]),

    "NP": dict(name="Nepal",           continent="Asia",
               renewable=98, policy=58, grid=55, tariff=0,
               chargers=62,   population=30, gdp_per_cap=1200,
               corridors=[]),

    "TW": dict(name="Taiwan",          continent="Asia",
               renewable=8,  policy=82, grid=99, tariff=0,
               chargers=18000, population=24, gdp_per_cap=33000,
               corridors=[]),

    "HK": dict(name="Hong Kong",       continent="Asia",
               renewable=1,  policy=72, grid=100, tariff=0,
               chargers=4200, population=7,  gdp_per_cap=50000,
               corridors=[]),

    "MN": dict(name="Mongolia",        continent="Asia",
               renewable=8,  policy=42, grid=52, tariff=15,
               chargers=28,   population=3,  gdp_per_cap=4200,
               corridors=[]),

    "KZ": dict(name="Kazakhstan",      continent="Asia",
               renewable=5,  policy=50, grid=72, tariff=20,
               chargers=82,   population=19, gdp_per_cap=10200,
               corridors=[]),

    "UZ": dict(name="Uzbekistan",      continent="Asia",
               renewable=8,  policy=45, grid=68, tariff=15,
               chargers=42,   population=36, gdp_per_cap=2100,
               corridors=[]),

    # ══════════════════════════════════════════════════════════
    # AMERICAS (35 countries)
    # ══════════════════════════════════════════════════════════

    "US": dict(name="United States",   continent="Americas",
               renewable=22, policy=72, grid=92, tariff=0,
               chargers=168000, population=335, gdp_per_cap=76000,
               corridors=[
                   dict(name="LA → San Francisco",   km=600,  existing=280, needed=15, gap=0),
                   dict(name="New York → Boston",     km=340,  existing=120, needed=9,  gap=0),
                   dict(name="Chicago → Detroit",     km=460,  existing=42,  needed=12, gap=8),
               ]),

    "CA": dict(name="Canada",          continent="Americas",
               renewable=68, policy=78, grid=94, tariff=0,
               chargers=28000, population=38, gdp_per_cap=52000,
               corridors=[
                   dict(name="Toronto → Montreal",   km=540, existing=48, needed=14, gap=5),
                   dict(name="Vancouver → Calgary",  km=970, existing=22, needed=24, gap=42),
               ]),

    "MX": dict(name="Mexico",          continent="Americas",
               renewable=25, policy=45, grid=75, tariff=16,
               chargers=2800, population=130, gdp_per_cap=10800,
               corridors=[
                   dict(name="Mexico City → Guadalajara", km=540, existing=8,  needed=14, gap=80),
                   dict(name="Mexico City → Monterrey",   km=920, existing=4,  needed=23, gap=92),
               ]),

    "BR": dict(name="Brazil",          continent="Americas",
               renewable=85, policy=68, grid=78, tariff=12,
               chargers=8400, population=215, gdp_per_cap=8700,
               corridors=[
                   dict(name="São Paulo → Rio",      km=430, existing=28, needed=11, gap=8),
                   dict(name="São Paulo → Curitiba", km=410, existing=18, needed=10, gap=18),
                   dict(name="Rio → Belo Horizonte", km=440, existing=12, needed=11, gap=32),
               ]),

    "AR": dict(name="Argentina",       continent="Americas",
               renewable=38, policy=52, grid=72, tariff=35,
               chargers=1200, population=46, gdp_per_cap=13000,
               corridors=[]),

    "CL": dict(name="Chile",           continent="Americas",
               renewable=42, policy=65, grid=82, tariff=5,
               chargers=1600, population=19, gdp_per_cap=16000,
               corridors=[
                   dict(name="Santiago → Valparaíso", km=120, existing=12, needed=3, gap=0),
                   dict(name="Santiago → Concepción", km=500, existing=6,  needed=13, gap=78),
               ]),

    "CO": dict(name="Colombia",        continent="Americas",
               renewable=68, policy=58, grid=72, tariff=15,
               chargers=480,  population=51, gdp_per_cap=6700,
               corridors=[
                   dict(name="Bogotá → Medellín",  km=420, existing=4, needed=11, gap=88),
               ]),

    "PE": dict(name="Peru",            continent="Americas",
               renewable=58, policy=45, grid=68, tariff=20,
               chargers=220,  population=33, gdp_per_cap=7400,
               corridors=[]),

    "EC": dict(name="Ecuador",         continent="Americas",
               renewable=72, policy=50, grid=72, tariff=15,
               chargers=180,  population=18, gdp_per_cap=6200,
               corridors=[]),

    "UY": dict(name="Uruguay",         continent="Americas",
               renewable=92, policy=72, grid=88, tariff=5,
               chargers=280,  population=3,  gdp_per_cap=17000,
               corridors=[]),

    "PY": dict(name="Paraguay",        continent="Americas",
               renewable=100, policy=42, grid=72, tariff=10,
               chargers=42,   population=7,  gdp_per_cap=5800,
               corridors=[]),

    "GT": dict(name="Guatemala",       continent="Americas",
               renewable=68, policy=35, grid=65, tariff=15,
               chargers=48,   population=18, gdp_per_cap=4600,
               corridors=[]),

    "CR": dict(name="Costa Rica",      continent="Americas",
               renewable=98, policy=72, grid=88, tariff=0,
               chargers=320,  population=5,  gdp_per_cap=13000,
               corridors=[]),

    "PA": dict(name="Panama",          continent="Americas",
               renewable=65, policy=55, grid=82, tariff=5,
               chargers=180,  population=4,  gdp_per_cap=15000,
               corridors=[]),

    "JM": dict(name="Jamaica",         continent="Americas",
               renewable=12, policy=50, grid=75, tariff=20,
               chargers=28,   population=3,  gdp_per_cap=5500,
               corridors=[]),

    "TT": dict(name="Trinidad & Tobago", continent="Americas",
               renewable=0,  policy=42, grid=88, tariff=15,
               chargers=18,   population=1,  gdp_per_cap=17000,
               corridors=[]),

    # ══════════════════════════════════════════════════════════
    # REST OF WORLD (additional)
    # ══════════════════════════════════════════════════════════

    "RU": dict(name="Russia",          continent="Europe",
               renewable=18, policy=35, grid=85, tariff=15,
               chargers=4200, population=144, gdp_per_cap=12000,
               corridors=[]),

    "TR": dict(name="Turkey",          continent="Europe",
               renewable=45, policy=65, grid=88, tariff=10,
               chargers=8400, population=84, gdp_per_cap=11000,
               corridors=[]),

    "UA": dict(name="Ukraine",         continent="Europe",
               renewable=18, policy=58, grid=65, tariff=0,
               chargers=2200, population=44, gdp_per_cap=4100,
               corridors=[]),

    "GE": dict(name="Georgia",         continent="Europe",
               renewable=82, policy=52, grid=78, tariff=0,
               chargers=180,  population=4,  gdp_per_cap=6200,
               corridors=[]),

    "AM": dict(name="Armenia",         continent="Europe",
               renewable=35, policy=48, grid=82, tariff=0,
               chargers=82,   population=3,  gdp_per_cap=5500,
               corridors=[]),

    "AZ": dict(name="Azerbaijan",      continent="Europe",
               renewable=8,  policy=48, grid=80, tariff=10,
               chargers=120,  population=10, gdp_per_cap=6800,
               corridors=[]),
}


def score(c):
    """Calculate weighted EV readiness score for a country dict."""
    import_score = max(0, 100 - c.get('tariff', 25) * 2)
    return round(
        c['renewable'] * 0.30 +
        c['policy']    * 0.30 +
        c['grid']      * 0.25 +
        import_score   * 0.15,
        1
    )

def tier(s):
    if s >= 65: return "HIGH"
    if s >= 40: return "MEDIUM"
    return "LOW"

def get_country(code):
    """Return enriched country dict with score and tier."""
    c = COUNTRIES.get(code.upper())
    if not c:
        return None
    s = score(c)
    return {**c, "code": code.upper(), "score": s, "tier": tier(s)}

def all_countries():
    """Return all countries sorted by readiness score descending."""
    result = []
    for code, c in COUNTRIES.items():
        s = score(c)
        result.append({**c, "code": code, "score": s, "tier": tier(s)})
    return sorted(result, key=lambda x: x['score'], reverse=True)

def by_continent():
    """Return countries grouped by continent, sorted by score."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in all_countries():
        groups[c['continent']].append(c)
    return dict(groups)

def top_n(n=20):
    return all_countries()[:n]

def needs_investment(min_gap=70):
    """Countries with critical corridor gaps needing investment."""
    results = []
    for code, c in COUNTRIES.items():
        critical = [cor for cor in c.get('corridors', []) if cor.get('gap', 0) >= min_gap]
        if critical:
            s = score(c)
            results.append({
                "code": code, "name": c['name'], "score": s,
                "continent": c['continent'],
                "critical_corridors": critical,
                "total_chargers_needed": sum(cor['needed'] for cor in critical),
            })
    return sorted(results, key=lambda x: x['total_chargers_needed'], reverse=True)


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  ChargeSmart Global Intelligence Database")
    print(f"{'='*55}")
    print(f"  Total countries: {len(COUNTRIES)}")

    continents = {}
    for c in COUNTRIES.values():
        continents[c['continent']] = continents.get(c['continent'], 0) + 1
    for cont, n in sorted(continents.items()):
        print(f"  {cont:<15} {n} countries")

    print(f"\n  TOP 10 EV READINESS SCORES:")
    for i, c in enumerate(top_n(10), 1):
        print(f"  {i:2}. {c['name']:<22} {c['score']:5.1f}  [{c['tier']}]  {c['continent']}")

    print(f"\n  HIGHEST INVESTMENT NEED (corridor gaps):")
    for c in needs_investment()[:8]:
        print(f"  {c['name']:<22} {c['total_chargers_needed']:3} chargers needed  ({len(c['critical_corridors'])} critical corridors)")

    scores = [score(c) for c in COUNTRIES.values()]
    print(f"\n  Global avg readiness: {sum(scores)/len(scores):.1f}")
    print(f"  HIGH tier (≥65):      {sum(1 for s in scores if s >= 65)} countries")
    print(f"  MEDIUM tier (40-64):  {sum(1 for s in scores if 40 <= s < 65)} countries")
    print(f"  LOW tier (<40):       {sum(1 for s in scores if s < 40)} countries")
    print()
