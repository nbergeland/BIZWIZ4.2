# === ENHANCED MULTI-CITY CONFIGURATION SYSTEM ===
# Save this as: city_config.py
# This replaces your existing city_config.py with enhanced USA-wide support

import json
import os
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import yaml
import time

@dataclass
class CityBounds:
    """Geographic boundaries for a city"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    center_lat: float
    center_lon: float
    grid_spacing: float = 0.005
    
    def get_grid_points(self) -> List[Tuple[float, float]]:
        """Generate grid points for the city"""
        import numpy as np
        lats = np.arange(self.min_lat, self.max_lat, self.grid_spacing)
        lons = np.arange(self.min_lon, self.max_lon, self.grid_spacing)
        return [(lat, lon) for lat in lats for lon in lons]

@dataclass
class CityDemographics:
    """Expected demographic ranges for normalization"""
    typical_population_range: Tuple[int, int]
    typical_income_range: Tuple[int, int] 
    typical_age_range: Tuple[float, float]
    population_density_factor: float = 1.0

@dataclass
class CityMarketData:
    """Market-specific data and API configurations"""
    state_code: str
    county_name: str
    city_name_variations: List[str]
    rental_api_city_name: str
    major_universities: List[str]
    major_employers: List[str]
    
@dataclass 
class CityCompetitorData:
    """Competitor-specific search terms and market factors"""
    primary_competitor: str
    competitor_search_terms: List[str]
    market_saturation_factor: float
    fast_casual_preference_score: float

@dataclass
class CityConfiguration:
    """Complete city configuration"""
    city_id: str
    display_name: str
    bounds: CityBounds
    demographics: CityDemographics
    market_data: CityMarketData
    competitor_data: CityCompetitorData
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CityConfiguration':
        """Create from dictionary"""
        return cls(
            city_id=data['city_id'],
            display_name=data['display_name'],
            bounds=CityBounds(**data['bounds']),
            demographics=CityDemographics(**data['demographics']),
            market_data=CityMarketData(**data['market_data']),
            competitor_data=CityCompetitorData(**data['competitor_data'])
        )

class USACityDataLoader:
    """Loads data for all USA cities"""
    
    def __init__(self):
        self.usa_cities_data = None
        
    def load_usa_cities_from_csv(self) -> pd.DataFrame:
        """Load cities from CSV data source"""
        # This uses a reliable source of US cities data
        try:
            # Option 1: Use a comprehensive US cities dataset
            url = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"
            df = pd.read_csv(url)
            
            # Filter for continental US and mid-to-large cities
            continental_states = [
                'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY',
                'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
                'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
                'WV', 'WI', 'WY', 'DC'
            ]
            
            # Filter for continental US
            df = df[df['STATE_CODE'].isin(continental_states)]
            
            # Filter for mid-to-large cities (population > 50,000)
            df = df[df['POPULATION'] >= 50000]
            
            # Sort by population descending
            df = df.sort_values('POPULATION', ascending=False)
            
            return df
            
        except Exception as e:
            print(f"Error loading from URL: {e}")
            return self._create_manual_top_cities()
    
    def _create_manual_top_cities(self) -> pd.DataFrame:
        """Create manual list of top US cities as fallback"""
        top_cities = [
            # Major metropolitan areas and mid-to-large cities
            {'CITY': 'New York', 'STATE_CODE': 'NY', 'LATITUDE': 40.7128, 'LONGITUDE': -74.0060, 'POPULATION': 8336817, 'COUNTY': 'New York County'},
            {'CITY': 'Los Angeles', 'STATE_CODE': 'CA', 'LATITUDE': 34.0522, 'LONGITUDE': -118.2437, 'POPULATION': 3979576, 'COUNTY': 'Los Angeles County'},
            {'CITY': 'Chicago', 'STATE_CODE': 'IL', 'LATITUDE': 41.8781, 'LONGITUDE': -87.6298, 'POPULATION': 2693976, 'COUNTY': 'Cook County'},
            {'CITY': 'Houston', 'STATE_CODE': 'TX', 'LATITUDE': 29.7604, 'LONGITUDE': -95.3698, 'POPULATION': 2320268, 'COUNTY': 'Harris County'},
            {'CITY': 'Phoenix', 'STATE_CODE': 'AZ', 'LATITUDE': 33.4484, 'LONGITUDE': -112.0740, 'POPULATION': 1680992, 'COUNTY': 'Maricopa County'},
            {'CITY': 'Philadelphia', 'STATE_CODE': 'PA', 'LATITUDE': 39.9526, 'LONGITUDE': -75.1652, 'POPULATION': 1584064, 'COUNTY': 'Philadelphia County'},
            {'CITY': 'San Antonio', 'STATE_CODE': 'TX', 'LATITUDE': 29.4241, 'LONGITUDE': -98.4936, 'POPULATION': 1547253, 'COUNTY': 'Bexar County'},
            {'CITY': 'San Diego', 'STATE_CODE': 'CA', 'LATITUDE': 32.7157, 'LONGITUDE': -117.1611, 'POPULATION': 1423851, 'COUNTY': 'San Diego County'},
            {'CITY': 'Dallas', 'STATE_CODE': 'TX', 'LATITUDE': 32.7767, 'LONGITUDE': -96.7970, 'POPULATION': 1343573, 'COUNTY': 'Dallas County'},
            {'CITY': 'San Jose', 'STATE_CODE': 'CA', 'LATITUDE': 37.3382, 'LONGITUDE': -121.8863, 'POPULATION': 1021795, 'COUNTY': 'Santa Clara County'},
            # Add more cities here... (continue with top 200+ cities)
            {'CITY': 'Austin', 'STATE_CODE': 'TX', 'LATITUDE': 30.2672, 'LONGITUDE': -97.7431, 'POPULATION': 978908, 'COUNTY': 'Travis County'},
            {'CITY': 'Jacksonville', 'STATE_CODE': 'FL', 'LATITUDE': 30.3322, 'LONGITUDE': -81.6557, 'POPULATION': 911507, 'COUNTY': 'Duval County'},
            {'CITY': 'Fort Worth', 'STATE_CODE': 'TX', 'LATITUDE': 32.7555, 'LONGITUDE': -97.3308, 'POPULATION': 909585, 'COUNTY': 'Tarrant County'},
            {'CITY': 'Columbus', 'STATE_CODE': 'OH', 'LATITUDE': 39.9612, 'LONGITUDE': -82.9988, 'POPULATION': 898553, 'COUNTY': 'Franklin County'},
            {'CITY': 'Charlotte', 'STATE_CODE': 'NC', 'LATITUDE': 35.2271, 'LONGITUDE': -80.8431, 'POPULATION': 885708, 'COUNTY': 'Mecklenburg County'},
            {'CITY': 'San Francisco', 'STATE_CODE': 'CA', 'LATITUDE': 37.7749, 'LONGITUDE': -122.4194, 'POPULATION': 881549, 'COUNTY': 'San Francisco County'},
            {'CITY': 'Indianapolis', 'STATE_CODE': 'IN', 'LATITUDE': 39.7684, 'LONGITUDE': -86.1581, 'POPULATION': 876384, 'COUNTY': 'Marion County'},
            {'CITY': 'Seattle', 'STATE_CODE': 'WA', 'LATITUDE': 47.6062, 'LONGITUDE': -122.3321, 'POPULATION': 753675, 'COUNTY': 'King County'},
            {'CITY': 'Denver', 'STATE_CODE': 'CO', 'LATITUDE': 39.7392, 'LONGITUDE': -104.9903, 'POPULATION': 715522, 'COUNTY': 'Denver County'},
            {'CITY': 'Washington', 'STATE_CODE': 'DC', 'LATITUDE': 38.9072, 'LONGITUDE': -77.0369, 'POPULATION': 705749, 'COUNTY': 'District of Columbia'},
            # Continue adding more cities...
        ]
        
        return pd.DataFrame(top_cities)

class EnhancedCityConfigManager:
    """Enhanced city configuration manager with USA-wide support"""
    
    def __init__(self, config_file: str = "usa_city_configs.yaml"):
        self.config_file = config_file
        self.configs: Dict[str, CityConfiguration] = {}
        self.current_city: Optional[str] = None
        self.data_loader = USACityDataLoader()
        self._setup_yaml()
        self.load_configs()
    
    def _setup_yaml(self):
        """Setup YAML to handle tuples properly"""
        def tuple_representer(dumper, data):
            return dumper.represent_list(list(data))
        
        def tuple_constructor(loader, node):
            return tuple(loader.construct_sequence(node))
        
        yaml.add_representer(tuple, tuple_representer)
        yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    
    def load_configs(self):
        """Load existing configs or create from USA cities data"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data and 'cities' in data:
                        for city_id, city_data in data['cities'].items():
                            self.configs[city_id] = CityConfiguration.from_dict(city_data)
                        self.current_city = data.get('current_city')
                        print(f"Loaded {len(self.configs)} city configurations")
                        return
            except Exception as e:
                print(f"Error loading existing configs: {e}")
        
        # Generate configs for USA cities
        print("Generating city configurations for USA cities...")
        self._generate_usa_city_configs()
    
    def _generate_usa_city_configs(self):
        """Generate configurations for all USA cities - FIXED VERSION"""
        print("Loading USA cities data...")
        
        # FIXED: Import the cities data from generate_usa_cities.py
        try:
            from generate_usa_cities import create_comprehensive_cities_list
            cities_df = create_comprehensive_cities_list()
            print(f"✅ Loaded {len(cities_df)} cities from generate_usa_cities.py")
        except ImportError as e:
            print(f"❌ Could not import generate_usa_cities: {e}")
            print("Falling back to external data source...")
            cities_df = self.data_loader.load_usa_cities_from_csv()
        
        print(f"Processing {len(cities_df)} cities...")
        
        # Define market data by state/region
        state_market_data = self._get_state_market_data()
        
        for idx, city_row in cities_df.iterrows():
            try:
                # Convert the row to match expected format if needed
                if 'city' in city_row:  # From generate_usa_cities.py format
                    city_data = {
                        'CITY': city_row['city'],
                        'STATE_CODE': city_row['state'], 
                        'LATITUDE': city_row['lat'],
                        'LONGITUDE': city_row['lon'],
                        'POPULATION': city_row['pop'],
                        'COUNTY': city_row['county']
                    }
                    city_series = pd.Series(city_data)
                else:  # Already in expected format
                    city_series = city_row
                
                city_config = self._create_city_config_from_data(city_series, state_market_data)
                self.configs[city_config.city_id] = city_config
                
                # Progress indicator
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(cities_df)} cities...")
                    
            except Exception as e:
                city_name = city_row.get('city', city_row.get('CITY', 'Unknown'))
                print(f"Error processing {city_name}: {e}")
                continue
        
        # Set default current city
        if not self.current_city and self.configs:
            self.current_city = next(iter(self.configs.keys()))
        
        print(f"✅ Generated {len(self.configs)} city configurations")
        self.save_configs()
    
    def _create_city_config_from_data(self, city_row: pd.Series, state_market_data: Dict) -> CityConfiguration:
        """Create a city configuration from data row"""
        city_name = city_row['CITY']
        state_code = city_row['STATE_CODE']
        population = city_row['POPULATION']
        lat = city_row['LATITUDE']
        lon = city_row['LONGITUDE']
        county = city_row.get('COUNTY', f"{city_name} County")
        
        # Create city ID
        city_id = f"{city_name.lower().replace(' ', '_').replace('-', '_').replace('.', '')}_{state_code.lower()}"
        
        # Calculate bounds based on population
        bounds_size = self._calculate_bounds_size(population)
        
        bounds = CityBounds(
            min_lat=lat - bounds_size,
            max_lat=lat + bounds_size,
            min_lon=lon - bounds_size,
            max_lon=lon + bounds_size,
            center_lat=lat,
            center_lon=lon,
            grid_spacing=0.005 if population > 500000 else 0.008
        )
        
        # Demographics based on population size
        demographics = self._create_demographics_for_population(population)
        
        # Market data
        state_info = state_market_data.get(state_code, {})
        market_data = CityMarketData(
            state_code=state_code,
            county_name=county,
            city_name_variations=[city_name, f"{city_name} {state_code}", f"{city_name}, {state_code}"],
            rental_api_city_name=city_name,
            major_universities=state_info.get('universities', []),
            major_employers=state_info.get('major_employers', [])
        )
        
        # Competitor data
        competitor_data = self._create_competitor_data_for_population(population)
        
        return CityConfiguration(
            city_id=city_id,
            display_name=f"{city_name}, {state_code}",
            bounds=bounds,
            demographics=demographics,
            market_data=market_data,
            competitor_data=competitor_data
        )
    
    def _calculate_bounds_size(self, population: int) -> float:
        """Calculate appropriate bounds size based on population"""
        if population > 2000000:
            return 0.3  # Large metro areas
        elif population > 1000000:
            return 0.2
        elif population > 500000:
            return 0.15
        elif population > 200000:
            return 0.1
        elif population > 100000:
            return 0.08
        else:
            return 0.05  # Smaller cities
    
    def _create_demographics_for_population(self, population: int) -> CityDemographics:
        """Create demographics based on population size"""
        if population > 1000000:
            return CityDemographics(
                typical_population_range=(10000, 50000),
                typical_income_range=(50000, 120000),
                typical_age_range=(25, 45),
                population_density_factor=1.5
            )
        elif population > 500000:
            return CityDemographics(
                typical_population_range=(5000, 25000),
                typical_income_range=(45000, 100000),
                typical_age_range=(25, 50),
                population_density_factor=1.2
            )
        elif population > 200000:
            return CityDemographics(
                typical_population_range=(3000, 15000),
                typical_income_range=(40000, 80000),
                typical_age_range=(25, 55),
                population_density_factor=1.0
            )
        else:
            return CityDemographics(
                typical_population_range=(2000, 10000),
                typical_income_range=(35000, 70000),
                typical_age_range=(25, 60),
                population_density_factor=0.8
            )
    
    def _create_competitor_data_for_population(self, population: int) -> CityCompetitorData:
        """Create competitor data based on population size"""
        base_competitors = ['mcdonalds', 'kfc', 'taco-bell', 'burger-king', 'subway', 'wendys', 'popeyes']
        
        if population > 1000000:
            # Large cities have more competition and saturation
            return CityCompetitorData(
                primary_competitor="chick-fil-a",
                competitor_search_terms=base_competitors + ['chipotle', 'panera', 'five-guys', 'shake-shack'],
                market_saturation_factor=0.95,
                fast_casual_preference_score=0.9
            )
        elif population > 500000:
            return CityCompetitorData(
                primary_competitor="chick-fil-a",
                competitor_search_terms=base_competitors + ['chipotle', 'panera'],
                market_saturation_factor=0.85,
                fast_casual_preference_score=0.85
            )
        elif population > 200000:
            return CityCompetitorData(
                primary_competitor="chick-fil-a",
                competitor_search_terms=base_competitors,
                market_saturation_factor=0.75,
                fast_casual_preference_score=0.8
            )
        else:
            return CityCompetitorData(
                primary_competitor="chick-fil-a",
                competitor_search_terms=base_competitors[:5],  # Fewer competitors in smaller cities
                market_saturation_factor=0.6,
                fast_casual_preference_score=0.75
            )
    
    def _get_state_market_data(self) -> Dict:
        """Get market data organized by state"""
        return {
            'CA': {
                'universities': ['UCLA', 'USC', 'Stanford', 'UC Berkeley'],
                'major_employers': ['Apple', 'Google', 'Disney', 'Tesla']
            },
            'TX': {
                'universities': ['UT Austin', 'Texas A&M', 'Rice University'],
                'major_employers': ['ExxonMobil', 'AT&T', 'Dell', 'Southwest Airlines']
            },
            'NY': {
                'universities': ['Columbia', 'NYU', 'Cornell'],
                'major_employers': ['JPMorgan Chase', 'Citigroup', 'IBM', 'Verizon']
            },
            'FL': {
                'universities': ['University of Florida', 'Florida State', 'Miami'],
                'major_employers': ['Disney', 'Publix', 'FedEx', 'NextEra Energy']
            },
            'ND': {
                'universities': ['University of North Dakota', 'North Dakota State University'],
                'major_employers': ['Sanford Health', 'Altru Health System', 'US Air Force']
            },
            # Add more states as needed...
        }
    
    def save_configs(self):
        """Save configurations to file"""
        data = {
            'current_city': self.current_city,
            'cities': {city_id: config.to_dict() for city_id, config in self.configs.items()}
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        
        print(f"Saved {len(self.configs)} city configurations to {self.config_file}")
    
    def get_cities_by_state(self, state_code: str) -> List[CityConfiguration]:
        """Get all cities for a specific state"""
        return [config for config in self.configs.values() 
                if config.market_data.state_code == state_code]
    
    def get_cities_by_population_range(self, min_pop: int, max_pop: int) -> List[CityConfiguration]:
        """Get cities within a population range"""
        return [config for config in self.configs.values() 
                if min_pop <= config.demographics.typical_population_range[1] <= max_pop]
    
    def search_cities(self, query: str) -> List[CityConfiguration]:
        """Search cities by name"""
        query_lower = query.lower()
        return [config for config in self.configs.values() 
                if query_lower in config.display_name.lower() or 
                query_lower in config.city_id.lower()]
    
    # Include all the original methods from CityConfigManager
    def set_current_city(self, city_id: str):
        """Set the current active city"""
        if city_id in self.configs:
            self.current_city = city_id
            self.save_configs()
            return True
        return False
    
    def get_current_config(self) -> Optional[CityConfiguration]:
        """Get the current city configuration"""
        if self.current_city and self.current_city in self.configs:
            return self.configs[self.current_city]
        return None
    
    def get_config(self, city_id: str) -> Optional[CityConfiguration]:
        """Get configuration for a specific city"""
        return self.configs.get(city_id)
    
    def list_cities(self) -> List[str]:
        """List available cities"""
        return list(self.configs.keys())

# Create alias for backward compatibility
CityConfigManager = EnhancedCityConfigManager

# === UTILITY FUNCTIONS FOR BACKWARD COMPATIBILITY ===

def get_city_bounds() -> Tuple[float, float, float, float]:
    """Get current city bounds for backward compatibility"""
    manager = CityConfigManager()
    config = manager.get_current_config()
    if config:
        return (config.bounds.min_lat, config.bounds.max_lat, 
                config.bounds.min_lon, config.bounds.max_lon)
    # Fallback to Grand Forks
    return (47.85, 47.95, -97.15, -97.0)

def get_grid_points() -> List[Tuple[float, float]]:
    """Get current city grid points for backward compatibility"""
    manager = CityConfigManager()
    config = manager.get_current_config()
    if config:
        return config.bounds.get_grid_points()
    # Fallback
    import numpy as np
    min_lat, max_lat, min_lon, max_lon = get_city_bounds()
    lats = np.arange(min_lat, max_lat, 0.005)
    lons = np.arange(min_lon, max_lon, 0.005)
    return [(lat, lon) for lat in lats for lon in lons]

def get_current_city_name() -> str:
    """Get current city display name"""
    manager = CityConfigManager()
    config = manager.get_current_config()
    return config.display_name if config else "Grand Forks, ND"

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    print("🏙️ Creating Enhanced USA City Configuration Manager...")
    
    # This will automatically load/generate configurations for all USA cities
    manager = CityConfigManager()  # Uses the alias for compatibility
    
    print(f"\n📊 Total cities configured: {len(manager.list_cities())}")
    
    # Show some statistics
    states = {}
    for config in manager.configs.values():
        state = config.market_data.state_code
        states[state] = states.get(state, 0) + 1
    
    print(f"📍 States covered: {len(states)}")
    print(f"🏆 Top 5 states by city count:")
    for state, count in sorted(states.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {state}: {count} cities")
    
    # Example searches
    print(f"\n🔍 Search examples:")
    texas_cities = manager.get_cities_by_state('TX')
    print(f"   Texas cities: {len(texas_cities)}")
    
    large_cities = manager.get_cities_by_population_range(500000, 10000000)
    print(f"   Large cities (500k+ population): {len(large_cities)}")
    
    chicago_results = manager.search_cities('chicago')
    print(f"   Cities matching 'chicago': {len(chicago_results)}")
    
    print(f"\n✅ Ready to use with your enhanced data collection system!")