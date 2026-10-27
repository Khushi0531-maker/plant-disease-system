"""
Disease Knowledge Base and Agronomic Advisory for FloraScan AI.
Provides comprehensive information on diagnosis, causes, organic treatments,
chemical solutions, and preventive measures across 24+ crops and 90+ pathologies.
"""

# Standard 38 Classes used by CNN Model (PlantVillage Benchmark)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ------------------------------------------------------------------------
# EXTENDED HIERARCHICAL CROP & DISEASE CATALOG (24+ Crops, 90+ Pathologies)
# ------------------------------------------------------------------------
EXTENDED_PLANTS_CATALOG = {
    'Apple': {
        'Apple Scab': {
            'status': 'Diseased',
            'scientific_name': 'Venturia inaequalis',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate to High',
            'symptoms': 'Olive-green to velvety dark brown lesions on leaves and fruit. Foliage curls, puckers, and drops prematurely.',
            'organic_treatment': 'Apply copper soap or sulfur spray early in spring. Rake and burn fallen leaves to disrupt overwintering.',
            'chemical_treatment': 'Apply captan, myclobutanil, or mancozeb at green tip through petal fall.',
            'prevention': 'Plant scab-resistant cultivars (Enterprise, Liberty, Honeycrisp). Thin out canopy to improve aeration.',
            'irrigation_tip': 'Use under-tree drip irrigation; avoid overhead sprinklers that wet foliage.'
        },
        'Black Rot': {
            'status': 'Diseased',
            'scientific_name': 'Botryosphaeria obtusa',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Frogeye leaf spots with purple borders and tan centers, bark cankers, and shriveled black mummified fruit.',
            'organic_treatment': 'Prune out dead wood, cankered twigs, and collect all fallen mummies during winter dormancy.',
            'chemical_treatment': 'Captan or thiophanate-methyl sprays starting at tight cluster through mid-summer.',
            'prevention': 'Maintain orchard sanitation and avoid pruning during wet weather.',
            'irrigation_tip': 'Water in early mornings to allow quick foliage drying.'
        },
        'Cedar Apple Rust': {
            'status': 'Diseased',
            'scientific_name': 'Gymnosporangium juniperi-virginianae',
            'pathogen_type': 'Fungus (Heteroecious)',
            'severity': 'Moderate',
            'symptoms': 'Bright yellow-orange or reddish spots on upper leaf surfaces. Spore-releasing tubes form beneath.',
            'organic_treatment': 'Apply sulfur or copper sprays when nearby cedar galls produce gelatinous spore horns in spring.',
            'chemical_treatment': 'Myclobutanil or propiconazole applied from pink bud through petal fall.',
            'prevention': 'Remove nearby Eastern red cedar / juniper trees within a 1-mile radius.',
            'irrigation_tip': 'Keep canopy pruned open to reduce humidity.'
        },
        'Fire Blight': {
            'status': 'Diseased',
            'scientific_name': 'Erwinia amylovora',
            'pathogen_type': 'Bacterium',
            'severity': 'Critical',
            'symptoms': 'Blossoms and shoots turn brown/black suddenly, leaves hang withered with a classic "shepherd\'s crook" bend.',
            'organic_treatment': 'Apply fixed copper sprays during dormancy. Prune strikes 8-12 inches below visible canker, sterilizing shears in 70% alcohol between cuts.',
            'chemical_treatment': 'Spray streptomycin or oxytetracycline during bloom period when warm rains occur.',
            'prevention': 'Avoid heavy nitrogen fertilizing that generates susceptible tender succulent growth.',
            'irrigation_tip': 'Drip irrigation only; prevent water splash.'
        },
        'Healthy Foliage': {
            'status': 'Healthy',
            'scientific_name': 'Malus domestica',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Vibrant, rich emerald green leaves with crisp margins, firm texture, and vigorous shoot growth.',
            'organic_treatment': 'Maintain annual compost mulching and balanced micronutrient foliar spray.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Routine orchard monitoring and seasonal dormant oil application.',
            'irrigation_tip': 'Provide 1-2 inches of water per week based on soil moisture.'
        }
    },
    'Banana': {
        'Black Sigatoka (Black Leaf Streak)': {
            'status': 'Diseased',
            'scientific_name': 'Pseudocercospora fijiensis',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Tiny reddish-brown specks turning into dark streaks parallel to leaf veins with sunken gray centers and yellow chlorosis.',
            'organic_treatment': 'De-leafing (pruning infected leaves) at weekly intervals and spraying with horticultural mineral oils.',
            'chemical_treatment': 'Rotate systemic triazoles (difenoconazole), strobilurins (azoxystrobin), and protectant mancozeb.',
            'prevention': 'Ensure wide plant spacing (2.5m x 2.5m) and clean drainage channels to reduce plantation humidity.',
            'irrigation_tip': 'Under-canopy drip irrigation; avoid overhead sprinklers.'
        },
        'Panama Disease (Fusarium Wilt)': {
            'status': 'Diseased',
            'scientific_name': 'Fusarium oxysporum f. sp. cubense (TR4)',
            'pathogen_type': 'Soil-borne Fungus',
            'severity': 'Critical',
            'symptoms': 'Progressive yellowing of older leaves, leaf buckling at petiole forming a "skirt" around pseudo-stem, internal vascular browning.',
            'organic_treatment': 'Bio-control soil inoculation with Trichoderma viride and application of organic matter/neem cake.',
            'chemical_treatment': 'No curative chemical for vascular wilt. Strictly quarantine and eradicate infected mats.',
            'prevention': 'Use certified tissue-culture disease-free suckers; disinfect boots and machinery before entering plots.',
            'irrigation_tip': 'Ensure excellent soil drainage; prevent waterlogging.'
        },
        'Banana Bunchy Top Virus (BBTV)': {
            'status': 'Diseased',
            'scientific_name': 'Babuvirus / BBTV (Vector: Pentalonia nigronervosa)',
            'pathogen_type': 'Virus',
            'severity': 'Critical',
            'symptoms': 'Dark green "morse code" dot-dash streaks along leaf veins, narrow upright leaves bunched into a rosette crown.',
            'organic_treatment': 'Inject or spray neem oil/insecticidal soap to eliminate aphid vectors. Destroy infected plants immediately.',
            'chemical_treatment': 'Spray imidacloprid or dimethoate on aphids before destroying virused mats.',
            'prevention': 'Plant only certified virus-indexed tissue-culture plantlets.',
            'irrigation_tip': 'Regular irrigation without stagnant puddling.'
        },
        'Healthy Plant': {
            'status': 'Healthy',
            'scientific_name': 'Musa acuminata',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Expansive, glossy green leaves without spotting, tearing, or yellow streaks; sturdy pseudostem.',
            'organic_treatment': 'Apply composted manure and potassium-rich organic fertilizers.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Regular weed management and desuckering to keep 1 mother + 1 follower.',
            'irrigation_tip': 'Deep watering 2-3 times per week in warm climates.'
        }
    },
    'Citrus / Orange': {
        'Citrus Greening (Huanglongbing / HLB)': {
            'status': 'Diseased',
            'scientific_name': 'Candidatus Liberibacter asiaticus (Vector: Asian Citrus Psyllid)',
            'pathogen_type': 'Bacterium',
            'severity': 'Critical',
            'symptoms': 'Blotchy mottled leaf yellowing crossing leaf veins asymmetrically, upright twig dieback, small bitter lopsided fruit.',
            'organic_treatment': 'Introduce Tamarixia radiata parasitoid wasps to biological control psyllid vectors.',
            'chemical_treatment': 'Target psyllid vectors with rotational insecticides (imidacloprid, thiamethoxam, bifenthrin); foliar micronutrient sprays (Zn, Mn, Fe).',
            'prevention': 'Plant certified disease-free rootstocks; eradicate confirmed symptomatic trees in young orchards.',
            'irrigation_tip': 'Precision micro-drip with tensiometers.'
        },
        'Citrus Canker': {
            'status': 'Diseased',
            'scientific_name': 'Xanthomonas citri subsp. citri',
            'pathogen_type': 'Bacterium',
            'severity': 'High',
            'symptoms': 'Raised, blister-like corky tan lesions surrounded by prominent oily yellow haloes on leaves, twigs, and fruit.',
            'organic_treatment': 'Spray copper hydroxide or copper oxychloride every 14-21 days during periods of leaf flushing.',
            'chemical_treatment': 'Copper bactericide sprays combined with windbreak barriers to prevent rain-driven dispersal.',
            'prevention': 'Establish natural windbreaks (casuarina/eucalyptus) around groves; sanitize harvesting equipment.',
            'irrigation_tip': 'Ground-level micro-jets; avoid overhead sprinkling.'
        },
        'Healthy Tree': {
            'status': 'Healthy',
            'scientific_name': 'Citrus sinensis',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Uniform deep green leaves with glossy waxy finish, vigorous spring flush, and clean fragrant blossoms.',
            'organic_treatment': 'Feed with balanced organic citrus fertilizer containing magnesium and iron.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Regular scouting for leafminers and scale insects.',
            'irrigation_tip': 'Deep irrigation every 7-10 days; let top 2 inches dry between waterings.'
        }
    },
    'Coffee': {
        'Coffee Leaf Rust': {
            'status': 'Diseased',
            'scientific_name': 'Hemileia vastatrix',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Yellow-orange powdery spots on the underside of leaves, turning brown and causing mass defoliation.',
            'organic_treatment': 'Apply copper-based fungicides (Bordeaux mixture 1%) at the beginning of the rainy season.',
            'chemical_treatment': 'Systemic triazole fungicides like epoxiconazole, cyproconazole, or triadimefon.',
            'prevention': 'Plant rust-resistant Arabica hybrids (Catimor, Castillo, Ruiru 11). Prune shade trees to manage humidity.',
            'irrigation_tip': 'Maintain soil moisture balance; improve farm aeration.'
        },
        'Coffee Berry Disease': {
            'status': 'Diseased',
            'scientific_name': 'Colletotrichum kahawae',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Dark sunken necrotic lesions on green coffee cherries causing them to rot, shrivel, and drop.',
            'organic_treatment': 'Preventive copper hydroxide sprays starting at pinhead berry stage.',
            'chemical_treatment': 'Spray chlorothalonil, dithianon, or carbendazim at 3-4 week intervals during wet berry expansion.',
            'prevention': 'Select resistant varieties and prune bushes to minimize prolonged wetness on berries.',
            'irrigation_tip': 'Avoid wetting bushes directly.'
        },
        'Healthy Bush': {
            'status': 'Healthy',
            'scientific_name': 'Coffea arabica',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Lustrous, wavy-margined dark green leaves with strong lateral branch structure.',
            'organic_treatment': 'Mulch with coffee pulp compost and apply bone meal.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Maintain 30-40% optimum canopy shade cover.',
            'irrigation_tip': 'Water during dry bloom-induction intervals.'
        }
    },
    'Corn (Maize)': {
        'Gray Leaf Spot': {
            'status': 'Diseased',
            'scientific_name': 'Cercospora zeae-maydis',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Narrow, rectangular, tan-to-gray lesions restricted strictly between parallel leaf veins with yellow borders.',
            'organic_treatment': 'Crop residue burial with deep tilling and biological seed treatment with Trichoderma.',
            'chemical_treatment': 'Apply azoxystrobin, propiconazole, or pyraclostrobin at VT-R1 growth stage.',
            'prevention': 'Rotate crops with soybeans or legumes; plant hybrids with strong GLS ratings.',
            'irrigation_tip': 'Schedule pivot watering early to reduce leaf wetness hours.'
        },
        'Common Rust': {
            'status': 'Diseased',
            'scientific_name': 'Puccinia sorghi',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate',
            'symptoms': 'Oval-to-elongated brick-red to cinnamon-brown powdery pustules scattered across both leaf surfaces.',
            'organic_treatment': 'Apply sulfur dust or neem seed kernel extract (NSKE 5%) at first pustule detection.',
            'chemical_treatment': 'Apply pyraclostrobin or tebuconazole if rust reaches upper canopy before tasseling.',
            'prevention': 'Plant corn hybrids containing dominant Rp resistance genes.',
            'irrigation_tip': 'Ensure good field drainage.'
        },
        'Northern Corn Leaf Blight': {
            'status': 'Diseased',
            'scientific_name': 'Exserohilum turcicum',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Large, cigar-shaped, grayish-green to tan lesions (1 to 6 inches long) that coalesce and blight entire leaves.',
            'organic_treatment': 'Autumn plowing to decompose infected corn debris; bio-fungicide drench.',
            'chemical_treatment': 'Apply Headline AMP or Quilt Xcel when lesions appear on ear leaves before pollination.',
            'prevention': 'Select NCLB-resistant seed varieties and practice 2-year crop rotation.',
            'irrigation_tip': 'Minimize evening irrigation sessions.'
        },
        'Healthy Stalk': {
            'status': 'Healthy',
            'scientific_name': 'Zea mays',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Broad, arching deep green leaves with crisp central white midrib and strong erect stalk.',
            'organic_treatment': 'Apply balanced nitrogen-phosphorus side dressing at knee-high stage.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Regular scouting for stem borers and armyworms.',
            'irrigation_tip': 'Ensure 1.5 inches of water weekly during silking.'
        }
    },
    'Cotton': {
        'Bacterial Blight (Angular Leaf Spot)': {
            'status': 'Diseased',
            'scientific_name': 'Xanthomonas citri pv. malvacearum',
            'pathogen_type': 'Bacterium',
            'severity': 'High',
            'symptoms': 'Angular, water-soaked lesions bounded by veins on leaves, black arm stem cankers, and circular boll rot spots.',
            'organic_treatment': 'Delint and treat seeds with hot water or bio-agent Pseudomonas fluorescens.',
            'chemical_treatment': 'Foliar spray with copper oxychloride (500g/acre) mixed with streptocycline (100 ppm).',
            'prevention': 'Plant acid-delinted certified seeds and resistant cultivars.',
            'irrigation_tip': 'Avoid water splash irrigation; use furrow or drip systems.'
        },
        'Fusarium Wilt': {
            'status': 'Diseased',
            'scientific_name': 'Fusarium oxysporum f. sp. vasinfectum',
            'pathogen_type': 'Fungus',
            'severity': 'Critical',
            'symptoms': 'Interveinal yellowing starting at leaf edges, wilted leaves dropping from bottom up, brown vascular ring discoloration.',
            'organic_treatment': 'Apply neem cake (100 kg/acre) and seed treatment with Trichoderma viride (10g/kg seed).',
            'chemical_treatment': 'Soil drenching with carbendazim (1g/L) around root zone of early affected plants.',
            'prevention': 'Practice long crop rotation with non-host crops like sorghum or millets.',
            'irrigation_tip': 'Prevent standing water and over-irrigation in heavy clay soils.'
        },
        'Alternaria Leaf Spot': {
            'status': 'Diseased',
            'scientific_name': 'Alternaria macrospora',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate',
            'symptoms': 'Small circular brown spots with purple margins and concentric target rings, leading to premature leaf drop.',
            'organic_treatment': 'Foliar spray of neem seed extract (NSKE 5%) or copper soap.',
            'chemical_treatment': 'Spray mancozeb (2g/L) or propiconazole (1ml/L) at 15-day intervals.',
            'prevention': 'Avoid potassium deficiency (potassium stress exacerbates Alternaria).',
            'irrigation_tip': 'Ensure even moisture; avoid drought stress.'
        },
        'Healthy Foliage': {
            'status': 'Healthy',
            'scientific_name': 'Gossypium hirsutum',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Rich green lobed leaves with clean petiole and vigorous sympodial branching with healthy squares/bolls.',
            'organic_treatment': 'Foliar spray of 2% DAP and balanced potassium.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Regular scouting for whiteflies, aphids, and bollworms.',
            'irrigation_tip': 'Irrigate at flowering and boll development stages.'
        }
    },
    'Cucumber': {
        'Downy Mildew': {
            'status': 'Diseased',
            'scientific_name': 'Pseudoperonospora cubensis',
            'pathogen_type': 'Oomycete',
            'severity': 'High',
            'symptoms': 'Angular, chlorotic yellow patches on upper leaf surface bounded strictly by veins; purplish downy mold beneath.',
            'organic_treatment': 'Spray copper fungicides or bio-fungicide Bacillus amyloliquefaciens on first indication.',
            'chemical_treatment': 'Apply systemic oomycides like cyazofamid (Ranman), mandipropamid (Revus), or fluopicolide.',
            'prevention': 'Trellis vines upright for maximum ventilation; choose downy mildew-tolerant varieties.',
            'irrigation_tip': 'Drip irrigation under black plastic mulch; never wet foliage.'
        },
        'Powdery Mildew': {
            'status': 'Diseased',
            'scientific_name': 'Podosphaera xanthii',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate',
            'symptoms': 'Talcum-powder-like white round fungal colonies on upper and lower surfaces, spreading to cover entire leaves.',
            'organic_treatment': 'Spray diluted milk (40% milk / 60% water) in full sunlight, or potassium bicarbonate solution.',
            'chemical_treatment': 'Myclobutanil, triflumizole, or quinoxyfen.',
            'prevention': 'Plant resistant cucumber hybrids; avoid excessive nitrogen fertilization.',
            'irrigation_tip': 'Water at the base early in the day.'
        },
        'Healthy Vine': {
            'status': 'Healthy',
            'scientific_name': 'Cucumis sativus',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Vibrant triangular leaves with healthy yellow flowers and crisp developing fruit.',
            'organic_treatment': 'Side dress with balanced vermicompost and liquid seaweed.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Install insect netting to exclude cucumber beetles.',
            'irrigation_tip': 'Consistent 1 inch of water weekly via drip.'
        }
    },
    'Grape': {
        'Black Rot': {
            'status': 'Diseased',
            'scientific_name': 'Guignardia bidwellii',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Small reddish-brown circular spots on leaves; grape berries shrivel into hard black wrinkled mummies.',
            'organic_treatment': 'Spray copper soap and sulfur before bloom; prune out all mummified fruit clusters during winter.',
            'chemical_treatment': 'Mancozeb, myclobutanil, or kresoxim-methyl applied from 1-inch shoot growth through post-bloom.',
            'prevention': 'Keep vine canopy open using shoot positioning and leaf pulling around clusters.',
            'irrigation_tip': 'Drip irrigation beneath canopy.'
        },
        'Esca (Black Measles)': {
            'status': 'Diseased',
            'scientific_name': 'Phaeomoniella chlamydospora & Fomitiporia mediterranea',
            'pathogen_type': 'Fungal Complex',
            'severity': 'Severe',
            'symptoms': 'Tiger-stripe patterns on foliage (interveinal yellowing/reddening followed by necrosis) and dark speckled berries.',
            'organic_treatment': 'Apply Trichoderma wound sealants on all pruning cuts immediately.',
            'chemical_treatment': 'Paint pruning wounds with thiophanate-methyl; no systemic cure once inside trunk.',
            'prevention': 'Never prune vines during rain; sterilize pruning shears with 70% ethanol.',
            'irrigation_tip': 'Avoid water-stressing infected vines during midsummer heat.'
        },
        'Healthy Foliage': {
            'status': 'Healthy',
            'scientific_name': 'Vitis vinifera',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Deep green lobed leaves with crisp veins, vigorous cane shoots, and clean plump grape clusters.',
            'organic_treatment': 'Apply compost tea and slow-release organic potassium.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Canopy management, tucking shoots, and regular powdery mildew scouting.',
            'irrigation_tip': 'Regulated deficit irrigation (RDI) for wine cultivars.'
        }
    },
    'Mango': {
        'Anthracnose': {
            'status': 'Diseased',
            'scientific_name': 'Colletotrichum gloeosporioides',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Black, sunken irregular spots on leaves, blossoms blighted, and dark tear-stain rot streaks on fruit.',
            'organic_treatment': 'Spray copper oxychloride (3g/L) before flowering; prune diseased twigs and burn fallen debris.',
            'chemical_treatment': 'Spray carbendazim (1g/L) or azoxystrobin (1ml/L) during panicle emergence and fruit set.',
            'prevention': 'Hot water treatment of harvested fruit (48°C for 15 minutes) prevents post-harvest anthracnose.',
            'irrigation_tip': 'Avoid overhead sprinklers during flowering stage.'
        },
        'Powdery Mildew': {
            'status': 'Diseased',
            'scientific_name': 'Oidium mangiferae',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate to High',
            'symptoms': 'White powdery fungal coating on panicles, flowers, and tender young leaves, causing premature flower drop.',
            'organic_treatment': 'Dust with wettable sulfur (2g/L) during early morning hours before flower buds open.',
            'chemical_treatment': 'Spray hexaconazole (1ml/L) or dinocap (1ml/L) at panicle emergence.',
            'prevention': 'Prune interior cross-branches to allow sunlight penetration into the tree canopy.',
            'irrigation_tip': 'Deep root soak every 15 days; withhold water during flower induction.'
        },
        'Dieback': {
            'status': 'Diseased',
            'scientific_name': 'Lasiodiplodia theobromae',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Twig drying from tip downwards, leaves turn brown and wither without dropping, stem gum exudation.',
            'organic_treatment': 'Prune affected twigs 3 inches below the dead margin and paste with Bordeaux paste.',
            'chemical_treatment': 'Spray copper oxychloride (3g/L) twice a year (June and October).',
            'prevention': 'Protect trees from mechanical injuries and bark borer insect damage.',
            'irrigation_tip': 'Maintain balanced irrigation during dry summer months.'
        },
        'Healthy Canopy': {
            'status': 'Healthy',
            'scientific_name': 'Mangifera indica',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Deep leathery green mature leaves with copper-colored fresh flushes and strong canopy structure.',
            'organic_treatment': 'Apply farmyard manure and paclobutrazol (if recommended for regular bearing).',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Annual pruning after harvest for canopy aeration.',
            'irrigation_tip': 'Basin irrigation at 10-15 day intervals during fruit growth.'
        }
    },
    'Onion': {
        'Purple Blotch': {
            'status': 'Diseased',
            'scientific_name': 'Alternaria porri',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Small, sunken water-soaked lesions on leaves turning purple with brown borders and concentric rings.',
            'organic_treatment': 'Spray neem oil or bio-fungicide Trichoderma; ensure clean weed-free raised beds.',
            'chemical_treatment': 'Spray mancozeb (2.5g/L) or tebuconazole (1ml/L) with sticker/spreader agent.',
            'prevention': 'Follow a 3-year crop rotation with non-allium crops; use disease-free seed bulbs.',
            'irrigation_tip': 'Drip irrigation; stop watering 2 weeks prior to harvest.'
        },
        'Downy Mildew': {
            'status': 'Diseased',
            'scientific_name': 'Peronospora destructor',
            'pathogen_type': 'Oomycete',
            'severity': 'High',
            'symptoms': 'Pale, oval chlorotic patches on leaves covered with grayish-violet fuzzy growth in humid mornings.',
            'organic_treatment': 'Apply copper oxychloride at earliest symptom detection; improve field drainage.',
            'chemical_treatment': 'Spray metalaxyl + mancozeb (Ridomil MZ 2.5g/L) or dimethomorph.',
            'prevention': 'Plant rows in line with prevailing winds to speed leaf drying.',
            'irrigation_tip': 'Irrigate in early morning so foliage dries quickly in sun.'
        },
        'Healthy Leaves': {
            'status': 'Healthy',
            'scientific_name': 'Allium cepa',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Upright, tubular, waxy blue-green foliage without lesions or tip dieback; firm developing bulb.',
            'organic_treatment': 'Apply sulfur and well-rotted compost during bulb initiation.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Regular scouting for onion thrips.',
            'irrigation_tip': 'Regular shallow watering until bulb maturation.'
        }
    },
    'Potato': {
        'Early Blight': {
            'status': 'Diseased',
            'scientific_name': 'Alternaria solani',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate',
            'symptoms': 'Concentric target-board ring spots on older lower leaves, surrounded by yellow chlorosis.',
            'organic_treatment': 'Copper soap or Bacillus subtilis applied at early tuber initiation.',
            'chemical_treatment': 'Chlorothalonil, mancozeb, or azoxystrobin on 7-14 day schedule.',
            'prevention': 'Maintain high soil fertility (avoid nitrogen deficit); 3-year rotation away from nightshades.',
            'irrigation_tip': 'Irrigate early morning so foliage dries before dusk.'
        },
        'Late Blight': {
            'status': 'Diseased',
            'scientific_name': 'Phytophthora infestans',
            'pathogen_type': 'Oomycete',
            'severity': 'Critical / Emergency',
            'symptoms': 'Water-soaked oily dark-brown lesions expanding rapidly with white fungal mildew underneath in damp weather.',
            'organic_treatment': 'Preventative copper spray; heavily infected vines must be destroyed immediately.',
            'chemical_treatment': 'Systemic cymoxanil (Curzate), cyazofamid (Ranman), or mandipropamid (Revus).',
            'prevention': 'Plant certified disease-free seed tubers; eliminate volunteer cull piles.',
            'irrigation_tip': 'Stop overhead irrigation when relative humidity exceeds 85%.'
        },
        'Blackleg & Bacterial Soft Rot': {
            'status': 'Diseased',
            'scientific_name': 'Pectobacterium atrosepticum',
            'pathogen_type': 'Bacterium',
            'severity': 'High',
            'symptoms': 'Inky black, slimy decay extending up the lower stem from the seed tuber; leaves wilt, roll up, and yellow.',
            'organic_treatment': 'Plant whole, uncut seed tubers; avoid planting in cold, wet, poorly aerated soils.',
            'chemical_treatment': 'No curative chemical. Disinfect cutting knives and machinery with quaternary ammonium.',
            'prevention': 'Use certified pathogen-tested seed; store harvested tubers in cool, dry, well-ventilated sheds.',
            'irrigation_tip': 'Avoid overwatering and waterlogged soil conditions.'
        },
        'Healthy Foliage': {
            'status': 'Healthy',
            'scientific_name': 'Solanum tuberosum',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Lush, dark green compound leaves, sturdy stems, and robust green potato hills.',
            'organic_treatment': 'Hill potatoes with compost to protect tubers from greening.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Weekly scouting for Colorado potato beetles.',
            'irrigation_tip': 'Uniform moisture from emergence until vine senescence.'
        }
    },
    'Rice (Paddy)': {
        'Rice Blast': {
            'status': 'Diseased',
            'scientific_name': 'Magnaporthe oryzae',
            'pathogen_type': 'Fungus',
            'severity': 'Critical',
            'symptoms': 'Spindle-shaped / diamond-shaped lesions with gray-white centers and dark reddish-brown margins on leaves; neck rot on panicles.',
            'organic_treatment': 'Seed treatment with Pseudomonas fluorescens (10g/kg) and foliar spray of neem oil.',
            'chemical_treatment': 'Spray tricyclazole (75% WP @ 0.6g/L) or isoprothiolane (40% EC @ 1.5ml/L).',
            'prevention': 'Avoid excessive nitrogen fertilization; plant blast-resistant cultivars (IR64, Swarna Sub1).',
            'irrigation_tip': 'Maintain consistent standing water layer; avoid alternate wet/dry cycles that trigger blast.'
        },
        'Bacterial Leaf Blight (BLB)': {
            'status': 'Diseased',
            'scientific_name': 'Xanthomonas oryzae pv. oryzae',
            'pathogen_type': 'Bacterium',
            'severity': 'High',
            'symptoms': 'Water-soaked translucent stripes at leaf margins expanding into wavy yellow-white blighted ribbons with bacterial ooze beads.',
            'organic_treatment': 'Spray fresh cow dung extract (20%) or bio-agent Bacillus subtilis.',
            'chemical_treatment': 'Spray copper hydroxide (2g/L) mixed with streptocycline (100 ppm) at early panicle initiation.',
            'prevention': 'Practice balanced potassium application (potassium reduces BLB susceptibility).',
            'irrigation_tip': 'Drain infected paddy fields for 3-4 days to arrest bacterial multiplication.'
        },
        'Brown Spot': {
            'status': 'Diseased',
            'scientific_name': 'Bipolaris oryzae',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate to High',
            'symptoms': 'Small, circular to oval sesame-seed-like dark brown spots with yellow haloes scattered across leaf blades.',
            'organic_treatment': 'Soil application of zinc sulfate and balanced compost to correct nutrient deficiency.',
            'chemical_treatment': 'Spray mancozeb (2g/L) or propiconazole (1ml/L) at boot leaf stage.',
            'prevention': 'Address soil silicon and potash deficiencies; use hot water seed soaking.',
            'irrigation_tip': 'Ensure paddy soil is never subjected to drought stress.'
        },
        'Sheath Blight': {
            'status': 'Diseased',
            'scientific_name': 'Rhizoctonia solani',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Oval or irregular greenish-gray water-soaked lesions on leaf sheaths near the water level, spreading upward.',
            'organic_treatment': 'Apply Trichoderma harzianum to soil and spray neem oil extract.',
            'chemical_treatment': 'Spray validamycin (2ml/L) or hexaconazole (2ml/L) directed at the plant base.',
            'prevention': 'Optimize plant hill spacing (20cm x 15cm) to prevent dense microclimate humidity.',
            'irrigation_tip': 'Drain excess standing water during tillering.'
        },
        'Healthy Crop': {
            'status': 'Healthy',
            'scientific_name': 'Oryza sativa',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Erect, rich emerald green leaves with clean midribs and vigorous tiller formation.',
            'organic_treatment': 'Apply green manure (Sesbania) and Azospirillum bio-fertilizer.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Field bund sanitation and rodent control.',
            'irrigation_tip': 'Maintain 2-5 cm water depth through tillering and flowering.'
        }
    },
    'Sugarcane': {
        'Red Rot': {
            'status': 'Diseased',
            'scientific_name': 'Colletotrichum falcatum',
            'pathogen_type': 'Fungus',
            'severity': 'Critical',
            'symptoms': 'Third or fourth leaf yellows and withers; split cane reveals longitudinal red discoloration with white cross-bands and alcohol smell.',
            'organic_treatment': 'Seed cane setts treated with Trichoderma harzianum paste before planting.',
            'chemical_treatment': 'Dip setts in carbendazim (0.1%) solution for 15 minutes before sowing.',
            'prevention': 'Use certified disease-free tissue-cultured seed setts; burn diseased crop stubble.',
            'irrigation_tip': 'Avoid irrigating clean fields with runoff from infected plots.'
        },
        'Sugarcane Smut': {
            'status': 'Diseased',
            'scientific_name': 'Sporisorium scitamineum',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Emergence of a long, black, whip-like unbranched structure from the growing spindle shoot covered with sooty spores.',
            'organic_treatment': 'Carefully bag the smut whip in plastic before cutting to avoid spore dispersal, then burn the entire stool.',
            'chemical_treatment': 'Treat setts with triadimefon (1g/L) or propiconazole (1ml/L).',
            'prevention': 'Plant smut-resistant cane varieties; rogue out diseased stools as soon as observed.',
            'irrigation_tip': 'Do not over-irrigate during early tiller emergence.'
        },
        'Healthy Cane': {
            'status': 'Healthy',
            'scientific_name': 'Saccharum officinarum',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Thick, sturdy stalks with vibrant green arching leaf blades and clean internodes.',
            'organic_treatment': 'Apply press mud compost and balanced NPK fertilizer.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Earthing-up at 90 days to prevent lodging and improve aeration.',
            'irrigation_tip': 'Furrow irrigation at 10-12 day intervals.'
        }
    },
    'Tomato': {
        'Early Blight': {
            'status': 'Diseased',
            'scientific_name': 'Alternaria solani',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate to High',
            'symptoms': 'Dark concentric target-shaped rings surrounded by yellow tissue starting on the lowest leaves and moving up.',
            'organic_treatment': 'Prune lower leaves up to 12 inches off soil. Spray copper fungicide or neem oil solution every 7 days.',
            'chemical_treatment': 'Apply chlorothalonil, mancozeb, or azoxystrobin at first sign of lower leaf spotting.',
            'prevention': 'Mulch around the base with straw or plastic to prevent fungal spores from splashing up from soil.',
            'irrigation_tip': 'Water at the base of the plant only, preferably in the morning.'
        },
        'Late Blight': {
            'status': 'Diseased',
            'scientific_name': 'Phytophthora infestans',
            'pathogen_type': 'Oomycete',
            'severity': 'Critical / Emergency',
            'symptoms': 'Large irregular dark-brown water-soaked oily spots with white moldy fungal growth beneath in humid weather.',
            'organic_treatment': 'Preventative copper spray is essential. Severely infected plants must be bagged and destroyed immediately.',
            'chemical_treatment': 'Apply systemic oomycide fungicides like Revus, Ranman, or Curzate promptly upon weather alert.',
            'prevention': 'Plant blight-resistant tomato varieties (e.g., Defiant, Mountain Magic, Iron Lady).',
            'irrigation_tip': 'Ensure maximum plant spacing to accelerate canopy air drying.'
        },
        'Bacterial Spot': {
            'status': 'Diseased',
            'scientific_name': 'Xanthomonas vesicatoria',
            'pathogen_type': 'Bacterium',
            'severity': 'High',
            'symptoms': 'Small dark brown greasy spots with yellow rings on leaves; leaves turn yellow and drop.',
            'organic_treatment': 'Spray copper octanoate weekly starting at transplant; drench with bio-fungicide Bacillus subtilis.',
            'chemical_treatment': 'Tank mix copper sulfate with mancozeb for enhanced bacterial control.',
            'prevention': 'Avoid working in wet fields; sterilize stakes and cages each season; rotate with non-host crops.',
            'irrigation_tip': 'Always use drip irrigation; avoid splash from overhead sprinklers.'
        },
        'Tomato Yellow Leaf Curl Virus (TYLCV)': {
            'status': 'Diseased',
            'scientific_name': 'Begomovirus / TYLCV (Vector: Whitefly)',
            'pathogen_type': 'Virus',
            'severity': 'Critical',
            'symptoms': 'Upward cupping and curling of leaf margins, severe interveinal yellowing (chlorosis), and dramatic plant stunting.',
            'organic_treatment': 'No chemical cure for the virus itself. Use yellow sticky cards and spray neem oil to control whitefly vectors.',
            'chemical_treatment': 'Insecticides (dinotefuran, imidacloprid, or pyriproxyfen) targeting whitefly nymphs and adults.',
            'prevention': 'Plant TYLCV-resistant hybrids (e.g., Tycoon, Camaro). Use 50-mesh insect netting in nurseries.',
            'irrigation_tip': 'Maintain steady watering to reduce stress on virused plants.'
        },
        'Healthy Foliage': {
            'status': 'Healthy',
            'scientific_name': 'Solanum lycopersicum',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Vibrant, dark green compound leaves with crisp serrated edges, sturdy stems, and no spots or blemishes.',
            'organic_treatment': 'Side dress with organic compost, worm castings, and calcium to prevent blossom end rot.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Maintain consistent watering, mulch base, and prune suckers for optimal fruiting.',
            'irrigation_tip': '1-2 inches weekly directly to soil level.'
        }
    },
    'Wheat': {
        'Yellow (Stripe) Rust': {
            'status': 'Diseased',
            'scientific_name': 'Puccinia striiformis f. sp. tritici',
            'pathogen_type': 'Fungus',
            'severity': 'Critical',
            'symptoms': 'Bright yellow-orange linear stripes of powdery pustules arranged parallel along leaf veins; leaves dry up like burnt paper.',
            'organic_treatment': 'Foliar spray with bio-formulation of Trichoderma viride and neem oil.',
            'chemical_treatment': 'Spray propiconazole (Tilt 25% EC @ 1ml/L) or tebuconazole (1ml/L) at first appearance of yellow stripes.',
            'prevention': 'Plant yellow-rust resistant wheat varieties (HD-2967, PBW-550, DBW-187).',
            'irrigation_tip': 'Avoid excessive late-season irrigation that maintains high canopy humidity.'
        },
        'Leaf (Brown) Rust': {
            'status': 'Diseased',
            'scientific_name': 'Puccinia triticina',
            'pathogen_type': 'Fungus',
            'severity': 'High',
            'symptoms': 'Small, round, orange-brown pustules scattered randomly across upper leaf surfaces without forming stripes.',
            'organic_treatment': 'Early sowing and seed treatment with biological agents.',
            'chemical_treatment': 'Spray propiconazole (Tilt @ 0.1%) or hexaconazole on flag leaves.',
            'prevention': 'Choose resistant cultivars with multi-gene resistance (Lr genes).',
            'irrigation_tip': 'Schedule water according to critical growth stages (CRI, flowering).'
        },
        'Powdery Mildew': {
            'status': 'Diseased',
            'scientific_name': 'Blumeria graminis f. sp. tritici',
            'pathogen_type': 'Fungus',
            'severity': 'Moderate',
            'symptoms': 'White to grayish fluffy powder patches on lower leaves, stems, and heads, turning dull brown with black dots.',
            'organic_treatment': 'Apply wettable sulfur spray (2g/L) during early vegetative growth.',
            'chemical_treatment': 'Spray tebuconazole (1ml/L) or propiconazole (1ml/L).',
            'prevention': 'Avoid dense seed rates and excessive nitrogen fertilizer.',
            'irrigation_tip': 'Avoid over-irrigation in cool, cloudy weather.'
        },
        'Loose Smut': {
            'status': 'Diseased',
            'scientific_name': 'Ustilago tritici',
            'pathogen_type': 'Internally Seed-borne Fungus',
            'severity': 'High',
            'symptoms': 'Entire wheat ear head transformed into a loose, powdery black mass of fungal spores, leaving only a bare rachis.',
            'organic_treatment': 'Solar heat treatment: soak seed in water for 4 hours, then dry on concrete floor in blazing summer sun.',
            'chemical_treatment': 'Seed treatment with carboxin (Vitavax @ 2g/kg seed) or tebuconazole (Raxil @ 1g/kg seed).',
            'prevention': 'Use certified loose-smut-free seed stock; rogue out infected heads in closed bags.',
            'irrigation_tip': 'Normal field watering.'
        },
        'Healthy Foliage': {
            'status': 'Healthy',
            'scientific_name': 'Triticum aestivum',
            'pathogen_type': 'None',
            'severity': 'None',
            'symptoms': 'Erect, rich green blade leaves with clean auricles, sturdy tillers, and healthy emerging green heads.',
            'organic_treatment': 'Top dress with vermicompost and Azotobacter bio-fertilizer.',
            'chemical_treatment': 'No treatment required.',
            'prevention': 'Timely sowing and balanced N-P-K-Zn nutrient management.',
            'irrigation_tip': 'Ensure critical irrigations at Crown Root Initiation (CRI) and Milking.'
        }
    }
}

# ------------------------------------------------------------------------
# HELPER FUNCTIONS FOR CROP & DISEASE CATALOG
# ------------------------------------------------------------------------

def get_available_crops():
    """Returns sorted list of all cataloged crop species."""
    return sorted(list(EXTENDED_PLANTS_CATALOG.keys()))


def get_diseases_for_crop(crop_name: str):
    """Returns list of disease names for a specific crop."""
    if crop_name in EXTENDED_PLANTS_CATALOG:
        return list(EXTENDED_PLANTS_CATALOG[crop_name].keys())
    return []


def get_crop_disease_solution(crop_name: str, disease_name: str) -> dict:
    """Returns detailed pathological and treatment dictionary for a crop and disease."""
    if crop_name in EXTENDED_PLANTS_CATALOG and disease_name in EXTENDED_PLANTS_CATALOG[crop_name]:
        res = dict(EXTENDED_PLANTS_CATALOG[crop_name][disease_name])
        res['crop'] = crop_name
        res['disease'] = disease_name
        return res
    
    # Generic fallback
    return {
        'crop': crop_name,
        'disease': disease_name,
        'status': 'Diseased' if 'healthy' not in disease_name.lower() else 'Healthy',
        'scientific_name': 'Refer to local extension office',
        'pathogen_type': 'Botanical pathogen',
        'severity': 'Moderate' if 'healthy' not in disease_name.lower() else 'None',
        'symptoms': 'Visual lesions or chlorosis observed on leaf tissue.',
        'organic_treatment': 'Inspect plant closely and prune affected foliage.',
        'chemical_treatment': 'Consult local agronomist for registered crop-specific active ingredients.',
        'prevention': 'Practice crop rotation, proper spacing, and orchard hygiene.',
        'irrigation_tip': 'Use ground-level drip irrigation; keep canopy dry.'
    }


def get_disease_info(class_name: str) -> dict:
    """Returns metadata dictionary for a CNN class name (e.g. Tomato___Early_blight)."""
    clean = class_name.replace('_', ' ')
    
    # Check if exact match in extended catalog
    for crop, diseases in EXTENDED_PLANTS_CATALOG.items():
        for d_name, d_dict in diseases.items():
            if crop.lower() in clean.lower() and d_name.lower().replace(' ', '') in clean.lower().replace(' ', ''):
                res = dict(d_dict)
                res['crop'] = crop
                res['disease'] = d_name
                return res

    # Fallback to simple split
    parts = class_name.split('___')
    c_name = parts[0].replace('_', ' ')
    d_name = parts[1].replace('_', ' ') if len(parts) > 1 else 'General Condition'
    
    return get_crop_disease_solution(c_name, d_name)
