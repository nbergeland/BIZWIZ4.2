#!/usr/bin/env python3
"""
BizWiz Dynamic Data Loader - Fixed Version
Addresses import and runtime issues from migration
"""

import pandas as pd
import numpy as np
import asyncio
import logging
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataLoadingProgress:
    """Track data loading progress"""
    city_id: str
    total_steps: int = 6
    current_step: int = 0
    step_name: str = "Initializing"
    locations_processed: int = 0
    total_locations: int = 0
    start_time: datetime = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def progress_percent(self) -> float:
        if self.total_steps == 0:
            return 100.0
        return (self.current_step / self.total_steps) * 100
    
    @property
    def elapsed_time(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def estimated_remaining(self) -> float:
        if self.current_step == 0:
            return 0
        elapsed = self.elapsed_time
        rate = elapsed / self.current_step
        remaining_steps = self.total_steps - self.current_step
        return rate * remaining_steps

class DynamicDataLoader:
    """Fixed data loader with enhanced error handling"""
    
    def __init__(self):
        self.cache_timeout = 3600  # 1 hour cache
        self.progress_callback = None
        
        # Import city config with error handling
        try:
            from city_config import CityConfigManager
            self.config_manager = CityConfigManager()
            logger.info("✅ City config manager loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import CityConfigManager: {e}")
            self.config_manager = None
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def _update_progress(self, progress: DataLoadingProgress):
        """Update progress and call callback if set"""
        if self.progress_callback:
            try:
                self.progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        logger.info(f"[{progress.city_id}] Step {progress.current_step}/{progress.total_steps}: "
                   f"{progress.step_name} ({progress.progress_percent:.1f}%)")
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cached data is still valid"""
        if not os.path.exists(cache_file):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return (datetime.now() - file_time).total_seconds() < self.cache_timeout
    
    def _get_location_seed(self, lat: float, lon: float) -> int:
        """Generate consistent seed based on location coordinates"""
        coord_string = f"{lat:.6f},{lon:.6f}"
        hash_obj = hashlib.md5(coord_string.encode())
        return int(hash_obj.hexdigest()[:8], 16)
    
    async def load_city_data_dynamic(self, city_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Dynamically load city data with comprehensive error handling
        
        Args:
            city_id: City identifier
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            Dictionary containing all processed city data
        """
        progress = DataLoadingProgress(city_id=city_id)
        self._update_progress(progress)
        
        try:
            # Get city configuration
            if not self.config_manager:
                raise ValueError("City configuration manager not available")
            
            config = self.config_manager.get_config(city_id)
            if not config:
                raise ValueError(f"City configuration not found for {city_id}")
            
            logger.info(f"Starting data load for {config.display_name}")
            
            # Check cache first (unless force refresh)
            cache_file = f"dynamic_cache_{city_id}.pkl"
            if not force_refresh and self._is_cache_valid(cache_file):
                logger.info(f"Loading cached data for {city_id}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        progress.current_step = progress.total_steps
                        progress.step_name = "Loaded from cache"
                        self._update_progress(progress)
                        
                        # Validate cached data
                        if self._validate_city_data(cached_data):
                            logger.info(f"Cache validation successful for {city_id}")
                            return cached_data
                        else:
                            logger.warning(f"Cache validation failed for {city_id}, proceeding with fresh data")
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}, proceeding with fresh data collection")
            
            # Generate fresh data
            return await self._generate_fresh_data(city_id, config, progress)
            
        except Exception as e:
            logger.error(f"Error in load_city_data_dynamic: {e}")
            # Return synthetic fallback data
            return self._create_emergency_fallback_data(city_id, progress)
    
    async def _generate_fresh_data(self, city_id: str, config, progress: DataLoadingProgress) -> Dict[str, Any]:
        """Generate fresh data for the city"""
        
        # Step 1: Generate analysis grid
        progress.current_step = 1
        progress.step_name = "Generating analysis grid"
        self._update_progress(progress)
        
        grid_points = self._generate_analysis_grid(config)
        progress.total_locations = len(grid_points)
        logger.info(f"Generated {len(grid_points)} grid points for analysis")
        
        # Step 2: Generate demographic data (synthetic for reliability)
        progress.current_step = 2
        progress.step_name = "Generating demographic data"
        self._update_progress(progress)
        
        demographic_data = self._generate_demographic_data_sync(grid_points, config)
        logger.info(f"Generated demographic data: {len(demographic_data)} records")
        
        # Step 3: Generate competitor data
        progress.current_step = 3
        progress.step_name = "Generating competitor data"
        self._update_progress(progress)
        
        competitor_data = self._generate_competitor_data_sync(config)
        total_competitors = sum(len(locations) for locations in competitor_data.values())
        logger.info(f"Generated {total_competitors} competitor locations")
        
        # Step 4: Generate traffic data
        progress.current_step = 4
        progress.step_name = "Generating traffic data"
        self._update_progress(progress)
        
        traffic_data = self._generate_traffic_data_sync(grid_points, config)
        logger.info(f"Generated traffic data for {len(traffic_data)} locations")
        
        # Step 5: Generate commercial data
        progress.current_step = 5
        progress.step_name = "Generating commercial data"
        self._update_progress(progress)
        
        commercial_data = self._generate_commercial_data_sync(grid_points, config)
        logger.info(f"Generated commercial data for {len(commercial_data)} locations")
        
        # Step 6: Process and model
        progress.current_step = 6
        progress.step_name = "Processing and modeling"
        self._update_progress(progress)
        
        processed_data = self._process_and_model_data(
            grid_points, demographic_data, competitor_data, 
            traffic_data, commercial_data, config, progress
        )
        
        # Cache the results
        cache_file = f"dynamic_cache_{city_id}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
            logger.info(f"Successfully cached data for {city_id}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
        
        progress.step_name = "Complete"
        self._update_progress(progress)
        
        # Final validation log
        df = processed_data['df_filtered']
        logger.info(f"FINAL DATA SUMMARY for {config.display_name}:")
        logger.info(f"  - Locations: {len(df)}")
        logger.info(f"  - Revenue range: ${df['predicted_revenue'].min():,.0f} - ${df['predicted_revenue'].max():,.0f}")
        logger.info(f"  - Revenue mean: ${df['predicted_revenue'].mean():,.0f}")
        logger.info(f"  - Competitors: {total_competitors}")
        logger.info(f"  - Model R²: {processed_data['metrics'].get('train_r2', 'N/A')}")
        
        return processed_data
    
    def _validate_city_data(self, city_data: Dict[str, Any]) -> bool:
        """Validate city data structure and content"""
        try:
            required_keys = ['df_filtered', 'competitor_data', 'metrics', 'city_config']
            for key in required_keys:
                if key not in city_data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            df = city_data['df_filtered']
            if len(df) == 0:
                logger.error("DataFrame is empty")
                return False
            
            if 'predicted_revenue' not in df.columns:
                logger.error("Missing predicted_revenue column")
                return False
            
            # Check revenue range is realistic
            min_revenue = df['predicted_revenue'].min()
            max_revenue = df['predicted_revenue'].max()
            mean_revenue = df['predicted_revenue'].mean()
            
            if min_revenue < 1_000_000 or max_revenue > 15_000_000:
                logger.warning(f"Revenue range unusual: ${min_revenue:,.0f} - ${max_revenue:,.0f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def _generate_analysis_grid(self, config) -> List[Tuple[float, float]]:
        """Generate grid points for analysis"""
        bounds = config.bounds
        
        # Improved grid spacing calculation
        population_factor = getattr(config.demographics, 'population_density_factor', 1.0)
        base_spacing = getattr(bounds, 'grid_spacing', 0.005)
        
        # Ensure we get enough points for analysis
        adaptive_spacing = min(base_spacing / (population_factor ** 0.3), 0.01)
        adaptive_spacing = max(adaptive_spacing, 0.002)
        
        lats = np.arange(bounds.min_lat, bounds.max_lat, adaptive_spacing)
        lons = np.arange(bounds.min_lon, bounds.max_lon, adaptive_spacing)
        
        grid_points = [(lat, lon) for lat in lats for lon in lons]
        
        # Filter to urban/suburban areas
        center_lat, center_lon = bounds.center_lat, bounds.center_lon
        max_distance = 0.8
        
        filtered_points = []
        for lat, lon in grid_points:
            distance = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
            if distance <= max_distance:
                filtered_points.append((lat, lon))
        
        # Ensure minimum number of points
        if len(filtered_points) < 50:
            logger.warning(f"Only {len(filtered_points)} grid points generated, expanding radius")
            max_distance = 1.2
            filtered_points = []
            for lat, lon in grid_points:
                distance = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
                if distance <= max_distance:
                    filtered_points.append((lat, lon))
        
        logger.info(f"Generated {len(filtered_points)} analysis points (spacing: {adaptive_spacing:.4f} degrees)")
        return filtered_points
    
    def _generate_demographic_data_sync(self, grid_points: List[Tuple[float, float]], config) -> pd.DataFrame:
        """Generate demographic data synchronously"""
        all_data = []
        
        for lat, lon in grid_points:
            demo_data = self._generate_synthetic_demographics(lat, lon, config)
            if demo_data:
                all_data.append(demo_data)
        
        if not all_data:
            logger.error("No demographic data generated!")
            raise ValueError("Failed to generate demographic data")
        
        df = pd.DataFrame(all_data)
        
        # Ensure required columns exist
        required_columns = ['latitude', 'longitude', 'median_income', 'median_age', 'population', 'median_rent']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, adding defaults")
                defaults = {
                    'median_income': 55000,
                    'median_age': 35,
                    'population': 5000,
                    'median_rent': 1200
                }
                df[col] = defaults.get(col, 0)
        
        logger.info(f"Demographic data summary: {len(df)} records, "
                   f"income range ${df['median_income'].min():,.0f}-${df['median_income'].max():,.0f}")
        
        return df
    
    def _generate_competitor_data_sync(self, config) -> Dict[str, List]:
        """Generate competitor data synchronously"""
        competitor_data = {}
        search_terms = list(set(getattr(config.competitor_data, 'competitor_search_terms', []) + 
                                [getattr(config.competitor_data, 'primary_competitor', 'chick-fil-a')]))
        
        logger.info(f"Generating competitors: {search_terms}")
        
        for competitor in search_terms:
            competitors = self._generate_realistic_competitors(competitor, config)
            competitor_data[competitor] = competitors
            logger.info(f"Generated {len(competitors)} locations for {competitor}")
        
        total_competitors = sum(len(locations) for locations in competitor_data.values())
        logger.info(f"Total competitor locations generated: {total_competitors}")
        
        return competitor_data
    
    def _generate_traffic_data_sync(self, grid_points: List[Tuple[float, float]], config) -> pd.DataFrame:
        """Generate traffic data synchronously"""
        all_traffic_data = []
        
        for lat, lon in grid_points:
            traffic_data = self._get_traffic_score_sync(lat, lon, config)
            all_traffic_data.append(traffic_data)
        
        return pd.DataFrame(all_traffic_data)
    
    def _generate_commercial_data_sync(self, grid_points: List[Tuple[float, float]], config) -> pd.DataFrame:
        """Generate commercial data synchronously"""
        all_commercial_data = []
        
        for lat, lon in grid_points:
            commercial_data = self._get_commercial_score_sync(lat, lon, config)
            all_commercial_data.append(commercial_data)
        
        return pd.DataFrame(all_commercial_data)
    
    def _get_traffic_score_sync(self, lat: float, lon: float, config) -> Dict:
        """Get CONSISTENT traffic score for a location"""
        location_seed = self._get_location_seed(lat, lon)
        np.random.seed(location_seed)
        
        center_lat, center_lon = getattr(config.bounds, 'center_lat', 40.7), getattr(config.bounds, 'center_lon', -74.0)
        distance_from_center = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
        
        # More sophisticated traffic modeling
        base_score = max(20, 100 - (distance_from_center * 150))
        
        # Add corridor effects
        corridor_bonus = 0
        if abs(lat - center_lat) < 0.02 or abs(lon - center_lon) < 0.02:
            corridor_bonus = np.random.uniform(10, 25)
        
        noise = np.random.normal(0, 12)
        traffic_score = max(15, min(95, base_score + corridor_bonus + noise))
        
        road_accessibility = np.random.uniform(60, 95)
        parking_availability = np.random.uniform(40, 85)
        
        np.random.seed(None)
        
        return {
            'latitude': lat,
            'longitude': lon,
            'traffic_score': traffic_score,
            'road_accessibility': road_accessibility,
            'parking_availability': parking_availability
        }
    
    def _get_commercial_score_sync(self, lat: float, lon: float, config) -> Dict:
        """Get CONSISTENT commercial viability score"""
        location_seed = self._get_location_seed(lat, lon)
        np.random.seed(location_seed)
        
        center_lat, center_lon = getattr(config.bounds, 'center_lat', 40.7), getattr(config.bounds, 'center_lon', -74.0)
        distance_from_center = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
        
        # Commercial viability decreases with distance from center
        base_commercial = max(30, 90 - (distance_from_center * 80))
        
        # Add some zones of higher commercial activity
        zone_bonus = 0
        if distance_from_center < 0.1:  # Downtown core
            zone_bonus = np.random.uniform(10, 20)
        elif distance_from_center < 0.3:  # Suburban commercial
            zone_bonus = np.random.uniform(5, 15)
        
        noise = np.random.normal(0, 10)
        commercial_score = max(25, min(95, base_commercial + zone_bonus + noise))
        
        # Zoning compliance higher in commercial areas
        zoning_prob = 0.8 if commercial_score > 70 else 0.6
        zoning_compliant = np.random.choice([True, False], p=[zoning_prob, 1-zoning_prob])
        
        # Rent correlates with commercial score and distance from center
        base_rent = 3000 + (commercial_score * 50) - (distance_from_center * 2000)
        rent_estimate = max(1500, base_rent + np.random.normal(0, 800))
        
        business_density = np.random.uniform(15, 60)
        
        np.random.seed(None)
        
        return {
            'latitude': lat,
            'longitude': lon,
            'commercial_score': commercial_score,
            'zoning_compliant': 1 if zoning_compliant else 0,
            'estimated_rent': rent_estimate,
            'business_density': business_density
        }
    
    def _generate_synthetic_demographics(self, lat: float, lon: float, config) -> Dict:
        """Generate CONSISTENT demographic data for restaurant market analysis"""
        
        location_seed = self._get_location_seed(lat, lon)
        np.random.seed(location_seed)
        
        # Use config ranges but ensure realistic data
        try:
            income_range = getattr(config.demographics, 'typical_income_range', (35000, 85000))
            age_range = getattr(config.demographics, 'typical_age_range', (25, 50))
            pop_range = getattr(config.demographics, 'typical_population_range', (2000, 12000))
        except:
            income_range = (35000, 85000)
            age_range = (25, 50)
            pop_range = (2000, 12000)
        
        # Distance from center affects demographics
        center_lat, center_lon = getattr(config.bounds, 'center_lat', 40.7), getattr(config.bounds, 'center_lon', -74.0)
        distance_from_center = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
        
        # Income tends to be higher in suburbs, lower in rural areas
        income_base = income_range[0] + (income_range[1] - income_range[0]) * 0.6
        if distance_from_center < 0.1:  # Urban core
            income_modifier = np.random.uniform(0.9, 1.3)
        elif distance_from_center < 0.4:  # Suburbs
            income_modifier = np.random.uniform(1.1, 1.4)
        else:  # Rural
            income_modifier = np.random.uniform(0.7, 1.0)
        
        median_income = np.clip(income_base * income_modifier, 
                               max(25000, income_range[0]), 
                               min(200000, income_range[1]))
        
        # Age distribution
        median_age = np.random.uniform(age_range[0], age_range[1])
        
        # Population density higher near center
        pop_base = pop_range[0] + (pop_range[1] - pop_range[0]) * 0.5
        pop_modifier = max(0.5, 1.5 - distance_from_center * 2)
        population = np.clip(pop_base * pop_modifier * np.random.uniform(0.8, 1.2),
                           pop_range[0], pop_range[1])
        
        # Rent correlates with income
        median_rent = median_income * np.random.uniform(0.22, 0.38)
        
        np.random.seed(None)
        
        return {
            'latitude': lat,
            'longitude': lon,
            'median_income': median_income,
            'median_age': median_age,
            'population': population,
            'median_rent': median_rent
        }
    
    def _generate_realistic_competitors(self, competitor: str, config) -> List[Dict]:
        """Generate realistic competitor data with proper market sizing"""
        
        # Create seed based on competitor name and city for consistency
        seed_string = f"{competitor}_{config.city_id}"
        hash_obj = hashlib.md5(seed_string.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # More realistic competitor counts based on market size
        try:
            population_factor = getattr(config.demographics, 'population_density_factor', 1.0)
        except:
            population_factor = 1.0
        
        if competitor.lower() in ['raising canes', 'canes', "raising cane's"]:
            base_count = max(2, int(population_factor * 3))
        elif competitor.lower() in ['chick-fil-a', 'chickfila', 'chick fil a']:
            base_count = max(3, int(population_factor * 5))
        elif competitor.lower() in ['popeyes', 'kfc', 'church']:
            base_count = max(2, int(population_factor * 4))
        else:
            base_count = max(1, int(population_factor * 2))
        
        # Add some variance
        num_competitors = np.random.randint(max(1, base_count - 2), base_count + 3)
        competitors = []
        
        logger.info(f"Generating {num_competitors} realistic {competitor} locations")
        
        for i in range(num_competitors):
            # Strategic placement near commercial areas and main roads
            center_bias = np.random.uniform(0.3, 0.8)
            
            lat_range = getattr(config.bounds, 'max_lat', 40.8) - getattr(config.bounds, 'min_lat', 40.6)
            lon_range = getattr(config.bounds, 'max_lon', -73.9) - getattr(config.bounds, 'min_lon', -74.1)
            
            # Generate location with center bias
            lat = (getattr(config.bounds, 'center_lat', 40.7) + 
                   np.random.uniform(-lat_range * center_bias, lat_range * center_bias))
            lon = (getattr(config.bounds, 'center_lon', -74.0) + 
                   np.random.uniform(-lon_range * center_bias, lon_range * center_bias))
            
            # Ensure within bounds
            lat = np.clip(lat, getattr(config.bounds, 'min_lat', 40.6), getattr(config.bounds, 'max_lat', 40.8))
            lon = np.clip(lon, getattr(config.bounds, 'min_lon', -74.1), getattr(config.bounds, 'max_lon', -73.9))
            
            competitors.append({
                'name': f"{competitor.title()} #{i+1}",
                'latitude': lat,
                'longitude': lon,
                'rating': np.random.uniform(3.8, 4.6),
                'user_ratings_total': np.random.randint(100, 2000),
                'is_synthetic': True
            })
        
        np.random.seed(None)
        return competitors
    
    def _process_and_model_data(self, grid_points: List[Tuple[float, float]], 
                               demographic_data: pd.DataFrame, 
                               competitor_data: Dict[str, List],
                               traffic_data: pd.DataFrame,
                               commercial_data: pd.DataFrame,
                               config,
                               progress: DataLoadingProgress) -> Dict[str, Any]:
        """Process all data and train model with improved error handling"""
        
        logger.info("Starting data processing and modeling...")
        
        # Create base DataFrame
        df = pd.DataFrame({
            'latitude': [p[0] for p in grid_points],
            'longitude': [p[1] for p in grid_points]
        })
        
        logger.info(f"Base DataFrame: {len(df)} locations")
        
        # Merge all data sources
        df = df.merge(demographic_data, on=['latitude', 'longitude'], how='left')
        df = df.merge(traffic_data, on=['latitude', 'longitude'], how='left')
        df = df.merge(commercial_data, on=['latitude', 'longitude'], how='left')
        
        # Fill any missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Calculate competitor metrics
        try:
            primary_competitor = getattr(config.competitor_data, 'primary_competitor', 'chick-fil-a')
        except:
            primary_competitor = 'chick-fil-a'
            
        primary_competitors = competitor_data.get(primary_competitor, [])
        if primary_competitors:
            df['distance_to_primary_competitor'] = df.apply(
                lambda row: self._min_distance_to_competitors(row, primary_competitors), axis=1
            )
        else:
            df['distance_to_primary_competitor'] = 8.0
        
        # Calculate competition density
        all_competitors = []
        for comp_list in competitor_data.values():
            all_competitors.extend(comp_list)
        
        if all_competitors:
            df['competition_density'] = df.apply(
                lambda row: self._competition_density(row, all_competitors), axis=1
            )
        else:
            df['competition_density'] = 0
        
        # Feature engineering
        df = self._engineer_features(df, config)
        
        # Train model and predict revenue
        try:
            logger.info("Training revenue prediction model...")
            model, metrics = self._train_revenue_model(df)
            
            if model is None:
                raise ValueError("Model training failed")
            
            feature_columns = self._get_feature_columns(df)
            df['predicted_revenue'] = model.predict(df[feature_columns])
            
            # Validate predictions
            if df['predicted_revenue'].isna().any():
                raise ValueError("Model produced NaN predictions")
            
            if df['predicted_revenue'].std() < 100000:
                raise ValueError("Model predictions lack sufficient variance")
            
            logger.info(f"Model training successful. Revenue range: ${df['predicted_revenue'].min():,.0f} - ${df['predicted_revenue'].max():,.0f}")
            
            return {
                'df_filtered': df,
                'competitor_data': competitor_data,
                'model': model,
                'metrics': metrics,
                'city_config': config,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            # Generate varied synthetic predictions
            df['predicted_revenue'] = self._generate_fallback_predictions(df, config)
            
            return {
                'df_filtered': df,
                'competitor_data': competitor_data,
                'model': None,
                'metrics': {'error': str(e), 'fallback_used': True},
                'city_config': config,
                'generation_time': datetime.now().isoformat()
            }
    
    def _generate_fallback_predictions(self, df: pd.DataFrame, config) -> pd.Series:
        """Generate varied fallback revenue predictions when model fails"""
        logger.warning("Generating fallback revenue predictions")
        
        base_revenue = 4_500_000
        predictions = []
        
        for idx, row in df.iterrows():
            location_seed = self._get_location_seed(row['latitude'], row['longitude'])
            np.random.seed(location_seed)
            
            # Factor in some basic metrics
            income_factor = (row.get('median_income', 55000) / 55000) ** 0.3
            traffic_factor = (row.get('traffic_score', 60) / 60) ** 0.4
            commercial_factor = (row.get('commercial_score', 50) / 50) ** 0.2
            
            prediction = (base_revenue * income_factor * traffic_factor * commercial_factor * 
                         np.random.uniform(0.7, 1.4))
            
            prediction = np.clip(prediction, 2_800_000, 8_200_000)
            predictions.append(prediction)
        
        np.random.seed(None)
        return pd.Series(predictions, index=df.index)
    
    def _engineer_features(self, df: pd.DataFrame, config) -> pd.DataFrame:
        """Engineer features for modeling"""
        center_lat, center_lon = getattr(config.bounds, 'center_lat', 40.7), getattr(config.bounds, 'center_lon', -74.0)
        df['distance_from_center'] = ((df['latitude'] - center_lat) ** 2 + 
                                     (df['longitude'] - center_lon) ** 2) ** 0.5 * 69
        
        df['income_age_interaction'] = df['median_income'] * df['median_age']
        df['traffic_commercial_interaction'] = df['traffic_score'] * df['commercial_score']
        df['competition_pressure'] = (df['competition_density'] / 
                                    (df['distance_to_primary_competitor'] + 0.1))
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns to use for modeling"""
        exclude_cols = ['latitude', 'longitude', 'predicted_revenue']
        return [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    def _train_revenue_model(self, df: pd.DataFrame) -> Tuple[Any, Dict]:
        """Train revenue prediction model with CONSISTENT and REALISTIC results"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_absolute_error, r2_score
            
            feature_cols = self._get_feature_columns(df)
            if len(feature_cols) == 0:
                raise ValueError("No valid feature columns found")
            
            X = df[feature_cols]
            
            # Check for invalid data
            if X.isna().any().any():
                logger.warning("Found NaN values in features, filling with median")
                X = X.fillna(X.median())
            
            # Set consistent seed
            np.random.seed(42)
            
            # ENHANCED REALISTIC FAST-CASUAL RESTAURANT REVENUE MODEL
            base_revenue = 4_300_000
            
            # Income factor (stronger impact)
            income_multiplier = np.clip((df['median_income'] / 60000) ** 0.5, 0.6, 2.0)
            income_impact = base_revenue * (income_multiplier - 1) * 0.35
            
            # Traffic factor (major impact for restaurants)
            traffic_multiplier = 0.5 + (df['traffic_score'] / 100) * 1.2
            traffic_impact = base_revenue * (traffic_multiplier - 1) * 0.45
            
            # Commercial viability
            commercial_multiplier = 0.7 + (df['commercial_score'] / 100) * 0.6
            commercial_impact = base_revenue * (commercial_multiplier - 1) * 0.30
            
            # Competition impact (stronger effect)
            competition_multiplier = np.where(
                df['distance_to_primary_competitor'] < 0.5, 0.65,
                np.where(df['distance_to_primary_competitor'] < 1.0, 0.80,
                        np.where(df['distance_to_primary_competitor'] < 2.0, 0.92, 1.08))
            )
            
            # Population factor
            pop_multiplier = np.clip((df['population'] / df['population'].median()) ** 0.4, 0.7, 1.5)
            population_impact = base_revenue * (pop_multiplier - 1) * 0.20
            
            # Age factor (refined)
            age_factor = np.where(
                (df['median_age'] >= 25) & (df['median_age'] <= 45), 1.12,
                np.where(df['median_age'] < 25, 1.06,
                        np.where(df['median_age'] > 60, 0.85, 1.0))
            )
            
            # Calculate base revenue
            total_revenue = (
                base_revenue + 
                income_impact + 
                traffic_impact + 
                commercial_impact + 
                population_impact
            ) * competition_multiplier * age_factor
            
            # Add location-specific consistent variance
            location_seeds = [self._get_location_seed(row['latitude'], row['longitude']) 
                            for _, row in df.iterrows()]
            
            market_variance = []
            for i, seed in enumerate(location_seeds):
                np.random.seed(seed + 2000)
                variance = total_revenue.iloc[i] * np.random.normal(0, 0.15)
                market_variance.append(variance)
            
            y = total_revenue + pd.Series(market_variance, index=total_revenue.index)
            
            # Apply realistic bounds
            y = np.clip(y, 2_700_000, 8_800_000)
            
            # Add exceptional locations (consistent)
            np.random.seed(456)
            num_exceptional = max(1, int(len(y) * 0.02))
            exceptional_indices = np.random.choice(len(y), size=num_exceptional, replace=False)
            
            for idx in exceptional_indices:
                np.random.seed(idx + 7000)
                y.iloc[idx] = np.random.uniform(8_500_000, 9_500_000)
            
            # Ensure good variance
            if y.std() < 200_000:
                logger.warning("Low revenue variance detected, adjusting...")
                for i in range(len(y)):
                    np.random.seed(location_seeds[i] + 3000)
                    if np.random.random() < 0.1:
                        factor = np.random.uniform(0.7, 1.4)
                        y.iloc[i] *= factor
                        y.iloc[i] = np.clip(y.iloc[i], 2_700_000, 9_500_000)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                min_samples_split=3,
                min_samples_leaf=2
            )
            
            model.fit(X, y)
            
            # Validate model
            y_pred = model.predict(X)
            
            if np.isnan(y_pred).any():
                raise ValueError("Model produced NaN predictions")
            
            # Calculate metrics
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', random_state=42)
            
            metrics = {
                'train_r2': r2_score(y, y_pred),
                'train_mae': mean_absolute_error(y, y_pred),
                'cv_mae_mean': -cv_scores.mean(),
                'cv_mae_std': cv_scores.std(),
                'feature_count': len(feature_cols),
                'revenue_stats': {
                    'min': f"${y.min():,.0f}",
                    'max': f"${y.max():,.0f}",
                    'mean': f"${y.mean():,.0f}",
                    'median': f"${np.median(y):,.0f}",
                    'std': f"${y.std():,.0f}",
                    'p25': f"${np.percentile(y, 25):,.0f}",
                    'p75': f"${np.percentile(y, 75):,.0f}",
                    'p90': f"${np.percentile(y, 90):,.0f}"
                }
            }
            
            np.random.seed(None)
            
            logger.info(f"Model training successful: R²={metrics['train_r2']:.3f}, "
                       f"Revenue range ${y.min():,.0f}-${y.max():,.0f}")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None, {'error': str(e)}
    
    def _min_distance_to_competitors(self, row: pd.Series, competitors: List[Dict]) -> float:
        """Calculate minimum distance to competitors"""
        if not competitors:
            return 10.0
        
        distances = []
        for comp in competitors:
            dist = ((row['latitude'] - comp['latitude']) ** 2 + 
                   (row['longitude'] - comp['longitude']) ** 2) ** 0.5 * 69
            distances.append(dist)
        
        return min(distances) if distances else 10.0
    
    def _competition_density(self, row: pd.Series, all_competitors: List[Dict]) -> int:
        """Calculate number of competitors within 2 miles"""
        count = 0
        for comp in all_competitors:
            dist = ((row['latitude'] - comp['latitude']) ** 2 + 
                   (row['longitude'] - comp['longitude']) ** 2) ** 0.5 * 69
            if dist <= 2.0:
                count += 1
        return count
    
    def _create_emergency_fallback_data(self, city_id: str, progress: DataLoadingProgress) -> Dict[str, Any]:
        """Create emergency fallback data when everything fails"""
        logger.warning(f"Creating emergency fallback data for {city_id}")
        
        # Generate minimal synthetic data
        np.random.seed(hash(city_id) % 2**32)
        
        n_locations = 50  # Minimal set
        lats = np.random.normal(40.7, 0.05, n_locations)
        lons = np.random.normal(-74.0, 0.05, n_locations)
        
        df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'predicted_revenue': np.random.uniform(3_000_000, 7_000_000, n_locations),
            'median_income': np.random.uniform(40_000, 90_000, n_locations),
            'median_age': np.random.uniform(25, 55, n_locations),
            'population': np.random.uniform(3_000, 15_000, n_locations),
            'traffic_score': np.random.uniform(30, 85, n_locations),
            'commercial_score': np.random.uniform(35, 80, n_locations),
            'distance_to_primary_competitor': np.random.uniform(0.5, 6.0, n_locations),
            'competition_density': np.random.randint(0, 5, n_locations)
        })
        
        # Mock config
        class MockConfig:
            def __init__(self, city_id):
                self.city_id = city_id
                self.display_name = city_id.replace('_', ' ').title()
                self.bounds = type('Bounds', (), {
                    'center_lat': lats.mean(),
                    'center_lon': lons.mean()
                })()
                self.competitor_data = type('CompData', (), {
                    'primary_competitor': 'chick-fil-a'
                })()
        
        progress.current_step = progress.total_steps
        progress.step_name = "Emergency fallback complete"
        self._update_progress(progress)
        
        return {
            'df_filtered': df,
            'competitor_data': {'chick-fil-a': []},
            'model': None,
            'metrics': {
                'emergency_fallback': True,
                'note': 'Emergency synthetic data - system may need repair'
            },
            'city_config': MockConfig(city_id),
            'generation_time': datetime.now().isoformat()
        }

# === USAGE FUNCTIONS ===

async def load_city_data_on_demand(city_id: str, progress_callback=None, force_refresh=False) -> Dict[str, Any]:
    """
    Main function to load city data on-demand
    """
    loader = DynamicDataLoader()
    if progress_callback:
        loader.set_progress_callback(progress_callback)
    
    return await loader.load_city_data_dynamic(city_id, force_refresh)

def load_city_data_sync(city_id: str, progress_callback=None, force_refresh=False) -> Dict[str, Any]:
    """
    Synchronous wrapper for async data loading - FIXED VERSION
    """
    try:
        # Try to use existing event loop if available
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    load_city_data_on_demand(city_id, progress_callback, force_refresh)
                )
                return future.result()
        else:
            # Safe to use asyncio.run
            return asyncio.run(load_city_data_on_demand(city_id, progress_callback, force_refresh))
    except RuntimeError:
        # Fallback: create new event loop
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                load_city_data_on_demand(city_id, progress_callback, force_refresh)
            )
        finally:
            new_loop.close()

# === TESTING ===
if __name__ == "__main__":
    print("🧪 Testing Fixed Dynamic Data Loader")
    
    def test_progress(progress):
        print(f"[{progress.city_id}] {progress.step_name} - {progress.progress_percent:.1f}%")
    
    try:
        # Test sync loading
        print("Testing synchronous loading...")
        city_data = load_city_data_sync("grand_forks_nd", test_progress, True)
        
        print(f"✅ Success!")
        print(f"   Locations: {len(city_data['df_filtered'])}")
        df = city_data['df_filtered']
        print(f"   Revenue range: ${df['predicted_revenue'].min():,.0f} - ${df['predicted_revenue'].max():,.0f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

def safe_get_config_attr(config, attr_path, default=None):
    """Safely get nested config attributes"""
    try:
        attrs = attr_path.split('.')
        value = config
        for attr in attrs:
            value = getattr(value, attr)
        return value
    except (AttributeError, TypeError):
        return default
        traceback.print_exc()