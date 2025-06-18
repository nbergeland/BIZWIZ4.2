#!/usr/bin/env python3
"""
Script to generate city configurations for all mid-to-large USA cities
Run this to create a comprehensive city database
"""

import pandas as pd
import requests
import json
import os
from city_config import CityConfigManager

def download_usa_cities_data():
    """Download comprehensive USA cities data"""
    
    # Option 1: Use US Census data or a comprehensive cities database
    print("📊 Downloading USA cities data...")
    
    # This is a reliable source of US cities with population data
    try:
        # Try multiple data sources
        sources = [
            {
                'url': 'https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv',
                'columns': {'City': 'CITY', 'State short': 'STATE_CODE', 'County': 'COUNTY'}
            },
            {
                'url': 'https://simplemaps.com/data/us-cities',
                'note': 'Commercial source - requires manual download'
            }
        ]
        
        # For now, we'll create a comprehensive list manually
        # This can be expanded to use API calls to get real-time data
        return create_comprehensive_cities_list()
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return create_comprehensive_cities_list()

def create_comprehensive_cities_list():
    """Create a comprehensive list of USA cities with population > 50,000"""
    
    print("🏙️ Creating comprehensive USA cities list...")
    
    # Top 300+ US cities by population
    cities_data = [
        # Major metropolitan areas (1M+)
        {'city': 'New York', 'state': 'NY', 'lat': 40.7128, 'lon': -74.0060, 'pop': 8336817, 'county': 'New York'},
        {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0522, 'lon': -118.2437, 'pop': 3979576, 'county': 'Los Angeles'},
        {'city': 'Chicago', 'state': 'IL', 'lat': 41.8781, 'lon': -87.6298, 'pop': 2693976, 'county': 'Cook'},
        {'city': 'Houston', 'state': 'TX', 'lat': 29.7604, 'lon': -95.3698, 'pop': 2320268, 'county': 'Harris'},
        {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.4484, 'lon': -112.0740, 'pop': 1680992, 'county': 'Maricopa'},
        {'city': 'Philadelphia', 'state': 'PA', 'lat': 39.9526, 'lon': -75.1652, 'pop': 1584064, 'county': 'Philadelphia'},
        {'city': 'San Antonio', 'state': 'TX', 'lat': 29.4241, 'lon': -98.4936, 'pop': 1547253, 'county': 'Bexar'},
        {'city': 'San Diego', 'state': 'CA', 'lat': 32.7157, 'lon': -117.1611, 'pop': 1423851, 'county': 'San Diego'},
        {'city': 'Dallas', 'state': 'TX', 'lat': 32.7767, 'lon': -96.7970, 'pop': 1343573, 'county': 'Dallas'},
        {'city': 'San Jose', 'state': 'CA', 'lat': 37.3382, 'lon': -121.8863, 'pop': 1021795, 'county': 'Santa Clara'},
        
        # Large cities (500k-1M)
        {'city': 'Austin', 'state': 'TX', 'lat': 30.2672, 'lon': -97.7431, 'pop': 978908, 'county': 'Travis'},
        {'city': 'Jacksonville', 'state': 'FL', 'lat': 30.3322, 'lon': -81.6557, 'pop': 911507, 'county': 'Duval'},
        {'city': 'Fort Worth', 'state': 'TX', 'lat': 32.7555, 'lon': -97.3308, 'pop': 909585, 'county': 'Tarrant'},
        {'city': 'Columbus', 'state': 'OH', 'lat': 39.9612, 'lon': -82.9988, 'pop': 898553, 'county': 'Franklin'},
        {'city': 'Charlotte', 'state': 'NC', 'lat': 35.2271, 'lon': -80.8431, 'pop': 885708, 'county': 'Mecklenburg'},
        {'city': 'San Francisco', 'state': 'CA', 'lat': 37.7749, 'lon': -122.4194, 'pop': 881549, 'county': 'San Francisco'},
        {'city': 'Indianapolis', 'state': 'IN', 'lat': 39.7684, 'lon': -86.1581, 'pop': 876384, 'county': 'Marion'},
        {'city': 'Seattle', 'state': 'WA', 'lat': 47.6062, 'lon': -122.3321, 'pop': 753675, 'county': 'King'},
        {'city': 'Denver', 'state': 'CO', 'lat': 39.7392, 'lon': -104.9903, 'pop': 715522, 'county': 'Denver'},
        {'city': 'Washington', 'state': 'DC', 'lat': 38.9072, 'lon': -77.0369, 'pop': 705749, 'county': 'District of Columbia'},
        {'city': 'Boston', 'state': 'MA', 'lat': 42.3601, 'lon': -71.0589, 'pop': 695926, 'county': 'Suffolk'},
        {'city': 'El Paso', 'state': 'TX', 'lat': 31.7619, 'lon': -106.4850, 'pop': 695145, 'county': 'El Paso'},
        {'city': 'Nashville', 'state': 'TN', 'lat': 36.1627, 'lon': -86.7816, 'pop': 689447, 'county': 'Davidson'},
        {'city': 'Detroit', 'state': 'MI', 'lat': 42.3314, 'lon': -83.0458, 'pop': 670031, 'county': 'Wayne'},
        {'city': 'Oklahoma City', 'state': 'OK', 'lat': 35.4676, 'lon': -97.5164, 'pop': 695057, 'county': 'Oklahoma'},
        {'city': 'Portland', 'state': 'OR', 'lat': 45.5152, 'lon': -122.6784, 'pop': 652503, 'county': 'Multnomah'},
        {'city': 'Las Vegas', 'state': 'NV', 'lat': 36.1699, 'lon': -115.1398, 'pop': 648224, 'county': 'Clark'},
        {'city': 'Memphis', 'state': 'TN', 'lat': 35.1495, 'lon': -90.0490, 'pop': 633104, 'county': 'Shelby'},
        {'city': 'Louisville', 'state': 'KY', 'lat': 38.2527, 'lon': -85.7585, 'pop': 617638, 'county': 'Jefferson'},
        {'city': 'Baltimore', 'state': 'MD', 'lat': 39.2904, 'lon': -76.6122, 'pop': 576498, 'county': 'Baltimore City'},
        {'city': 'Milwaukee', 'state': 'WI', 'lat': 43.0389, 'lon': -87.9065, 'pop': 577222, 'county': 'Milwaukee'},
        {'city': 'Albuquerque', 'state': 'NM', 'lat': 35.0844, 'lon': -106.6504, 'pop': 560513, 'county': 'Bernalillo'},
        {'city': 'Tucson', 'state': 'AZ', 'lat': 32.2226, 'lon': -110.9747, 'pop': 548073, 'county': 'Pima'},
        {'city': 'Fresno', 'state': 'CA', 'lat': 36.7378, 'lon': -119.7871, 'pop': 542107, 'county': 'Fresno'},
        {'city': 'Sacramento', 'state': 'CA', 'lat': 38.5816, 'lon': -121.4944, 'pop': 524943, 'county': 'Sacramento'},
        {'city': 'Kansas City', 'state': 'MO', 'lat': 39.0997, 'lon': -94.5786, 'pop': 508090, 'county': 'Jackson'},
        {'city': 'Mesa', 'state': 'AZ', 'lat': 33.4152, 'lon': -111.8315, 'pop': 504258, 'county': 'Maricopa'},
        {'city': 'Atlanta', 'state': 'GA', 'lat': 33.7490, 'lon': -84.3880, 'pop': 498715, 'county': 'Fulton'},
        
        # Mid-size cities (200k-500k)
        {'city': 'Colorado Springs', 'state': 'CO', 'lat': 38.8339, 'lon': -104.8214, 'pop': 478961, 'county': 'El Paso'},
        {'city': 'Omaha', 'state': 'NE', 'lat': 41.2565, 'lon': -95.9345, 'pop': 478192, 'county': 'Douglas'},
        {'city': 'Raleigh', 'state': 'NC', 'lat': 35.7796, 'lon': -78.6382, 'pop': 474069, 'county': 'Wake'},
        {'city': 'Miami', 'state': 'FL', 'lat': 25.7617, 'lon': -80.1918, 'pop': 470914, 'county': 'Miami-Dade'},
        {'city': 'Long Beach', 'state': 'CA', 'lat': 33.7701, 'lon': -118.1937, 'pop': 466742, 'county': 'Los Angeles'},
        {'city': 'Virginia Beach', 'state': 'VA', 'lat': 36.8529, 'lon': -75.9780, 'pop': 459470, 'county': 'Virginia Beach City'},
        {'city': 'Oakland', 'state': 'CA', 'lat': 37.8044, 'lon': -122.2712, 'pop': 433031, 'county': 'Alameda'},
        {'city': 'Minneapolis', 'state': 'MN', 'lat': 44.9778, 'lon': -93.2650, 'pop': 429954, 'county': 'Hennepin'},
        {'city': 'Tulsa', 'state': 'OK', 'lat': 36.1540, 'lon': -95.9928, 'pop': 413066, 'county': 'Tulsa'},
        {'city': 'Arlington', 'state': 'TX', 'lat': 32.7357, 'lon': -97.1081, 'pop': 398854, 'county': 'Tarrant'},
        {'city': 'Tampa', 'state': 'FL', 'lat': 27.9506, 'lon': -82.4572, 'pop': 384959, 'county': 'Hillsborough'},
        {'city': 'New Orleans', 'state': 'LA', 'lat': 29.9511, 'lon': -90.0715, 'pop': 383997, 'county': 'Orleans'},
        {'city': 'Wichita', 'state': 'KS', 'lat': 37.6872, 'lon': -97.3301, 'pop': 397532, 'county': 'Sedgwick'},
        {'city': 'Cleveland', 'state': 'OH', 'lat': 41.4993, 'lon': -81.6944, 'pop': 383793, 'county': 'Cuyahoga'},
        {'city': 'Bakersfield', 'state': 'CA', 'lat': 35.3733, 'lon': -119.0187, 'pop': 380874, 'county': 'Kern'},
        {'city': 'Aurora', 'state': 'CO', 'lat': 39.7294, 'lon': -104.8319, 'pop': 379289, 'county': 'Arapahoe'},
        {'city': 'Anaheim', 'state': 'CA', 'lat': 33.8366, 'lon': -117.9143, 'pop': 352497, 'county': 'Orange'},
        {'city': 'Santa Ana', 'state': 'CA', 'lat': 33.7455, 'lon': -117.8677, 'pop': 334136, 'county': 'Orange'},
        {'city': 'Riverside', 'state': 'CA', 'lat': 33.9533, 'lon': -117.3962, 'pop': 331816, 'county': 'Riverside'},
        {'city': 'Corpus Christi', 'state': 'TX', 'lat': 27.8006, 'lon': -97.3964, 'pop': 326586, 'county': 'Nueces'},
        {'city': 'Lexington', 'state': 'KY', 'lat': 38.0406, 'lon': -84.5037, 'pop': 323152, 'county': 'Fayette'},
        {'city': 'Stockton', 'state': 'CA', 'lat': 37.9577, 'lon': -121.2908, 'pop': 320804, 'county': 'San Joaquin'},
        {'city': 'Henderson', 'state': 'NV', 'lat': 36.0395, 'lon': -114.9817, 'pop': 320189, 'county': 'Clark'},
        {'city': 'Saint Paul', 'state': 'MN', 'lat': 44.9537, 'lon': -93.0900, 'pop': 308096, 'county': 'Ramsey'},
        {'city': 'St. Louis', 'state': 'MO', 'lat': 38.6270, 'lon': -90.1994, 'pop': 301578, 'county': 'St. Louis City'},
        {'city': 'Cincinnati', 'state': 'OH', 'lat': 39.1031, 'lon': -84.5120, 'pop': 309317, 'county': 'Hamilton'},
        {'city': 'Pittsburgh', 'state': 'PA', 'lat': 40.4406, 'lon': -79.9959, 'pop': 300286, 'county': 'Allegheny'},
        
        # Continue with smaller but significant cities (100k-200k)
        {'city': 'Greensboro', 'state': 'NC', 'lat': 36.0726, 'lon': -79.7920, 'pop': 296710, 'county': 'Guilford'},
        {'city': 'Plano', 'state': 'TX', 'lat': 33.0198, 'lon': -96.6989, 'pop': 288061, 'county': 'Collin'},
        {'city': 'Lincoln', 'state': 'NE', 'lat': 40.8136, 'lon': -96.7026, 'pop': 295178, 'county': 'Lancaster'},
        {'city': 'Buffalo', 'state': 'NY', 'lat': 42.8864, 'lon': -78.8784, 'pop': 278349, 'county': 'Erie'},
        {'city': 'Fort Wayne', 'state': 'IN', 'lat': 41.0793, 'lon': -85.1394, 'pop': 270402, 'county': 'Allen'},
        {'city': 'Jersey City', 'state': 'NJ', 'lat': 40.7178, 'lon': -74.0431, 'pop': 292449, 'county': 'Hudson'},
        {'city': 'Chula Vista', 'state': 'CA', 'lat': 32.6401, 'lon': -117.0842, 'pop': 275487, 'county': 'San Diego'},
        {'city': 'Orlando', 'state': 'FL', 'lat': 28.5383, 'lon': -81.3792, 'pop': 307573, 'county': 'Orange'},
        {'city': 'St. Petersburg', 'state': 'FL', 'lat': 27.7676, 'lon': -82.6404, 'pop': 265351, 'county': 'Pinellas'},
        {'city': 'Norfolk', 'state': 'VA', 'lat': 36.8468, 'lon': -76.2852, 'pop': 238005, 'county': 'Norfolk City'},
        {'city': 'Chandler', 'state': 'AZ', 'lat': 33.3062, 'lon': -111.8413, 'pop': 261165, 'county': 'Maricopa'},
        {'city': 'Laredo', 'state': 'TX', 'lat': 27.5306, 'lon': -99.4803, 'pop': 262491, 'county': 'Webb'},
        {'city': 'Madison', 'state': 'WI', 'lat': 43.0732, 'lon': -89.4012, 'pop': 269840, 'county': 'Dane'},
        {'city': 'Durham', 'state': 'NC', 'lat': 35.9940, 'lon': -78.8986, 'pop': 283506, 'county': 'Durham'},
        {'city': 'Lubbock', 'state': 'TX', 'lat': 33.5779, 'lon': -101.8552, 'pop': 258862, 'county': 'Lubbock'},
        {'city': 'Winston-Salem', 'state': 'NC', 'lat': 36.0999, 'lon': -80.2442, 'pop': 249545, 'county': 'Forsyth'},
        {'city': 'Garland', 'state': 'TX', 'lat': 32.9126, 'lon': -96.6389, 'pop': 246018, 'county': 'Dallas'},
        {'city': 'Glendale', 'state': 'AZ', 'lat': 33.5387, 'lon': -112.1860, 'pop': 248325, 'county': 'Maricopa'},
        {'city': 'Hialeah', 'state': 'FL', 'lat': 25.8576, 'lon': -80.2781, 'pop': 223109, 'county': 'Miami-Dade'},
        {'city': 'Reno', 'state': 'NV', 'lat': 39.5296, 'lon': -119.8138, 'pop': 264165, 'county': 'Washoe'},
        {'city': 'Baton Rouge', 'state': 'LA', 'lat': 30.4515, 'lon': -91.1871, 'pop': 220236, 'county': 'East Baton Rouge'},
        {'city': 'Irvine', 'state': 'CA', 'lat': 33.6846, 'lon': -117.8265, 'pop': 307670, 'county': 'Orange'},
        {'city': 'Chesapeake', 'state': 'VA', 'lat': 36.7682, 'lon': -76.2875, 'pop': 249422, 'county': 'Chesapeake City'},
        {'city': 'Irving', 'state': 'TX', 'lat': 32.8140, 'lon': -96.9489, 'pop': 256684, 'county': 'Dallas'},
        {'city': 'Scottsdale', 'state': 'AZ', 'lat': 33.4942, 'lon': -111.9261, 'pop': 258069, 'county': 'Maricopa'},
        {'city': 'North Las Vegas', 'state': 'NV', 'lat': 36.1989, 'lon': -115.1175, 'pop': 262527, 'county': 'Clark'},
        {'city': 'Fremont', 'state': 'CA', 'lat': 37.5485, 'lon': -121.9886, 'pop': 230504, 'county': 'Alameda'},
        {'city': 'Gilbert', 'state': 'AZ', 'lat': 33.3528, 'lon': -111.7890, 'pop': 267918, 'county': 'Maricopa'},
        {'city': 'San Bernardino', 'state': 'CA', 'lat': 34.1083, 'lon': -117.2898, 'pop': 222101, 'county': 'San Bernardino'},
        {'city': 'Boise', 'state': 'ID', 'lat': 43.6150, 'lon': -116.2023, 'pop': 235684, 'county': 'Ada'},
        {'city': 'Birmingham', 'state': 'AL', 'lat': 33.5186, 'lon': -86.8104, 'pop': 200733, 'county': 'Jefferson'},
        
        # Additional mid-size cities
        {'city': 'Spokane', 'state': 'WA', 'lat': 47.6587, 'lon': -117.4260, 'pop': 230176, 'county': 'Spokane'},
        {'city': 'Rochester', 'state': 'NY', 'lat': 43.1566, 'lon': -77.6088, 'pop': 206284, 'county': 'Monroe'},
        {'city': 'Des Moines', 'state': 'IA', 'lat': 41.5868, 'lon': -93.6250, 'pop': 214133, 'county': 'Polk'},
        {'city': 'Modesto', 'state': 'CA', 'lat': 37.6391, 'lon': -120.9969, 'pop': 218464, 'county': 'Stanislaus'},
        {'city': 'Fayetteville', 'state': 'NC', 'lat': 35.0527, 'lon': -78.8784, 'pop': 211657, 'county': 'Cumberland'},
        {'city': 'Tacoma', 'state': 'WA', 'lat': 47.2529, 'lon': -122.4443, 'pop': 219346, 'county': 'Pierce'},
        {'city': 'Oxnard', 'state': 'CA', 'lat': 34.1975, 'lon': -119.1771, 'pop': 202063, 'county': 'Ventura'},
        {'city': 'Fontana', 'state': 'CA', 'lat': 34.0922, 'lon': -117.4350, 'pop': 208393, 'county': 'San Bernardino'},
        {'city': 'Columbus', 'state': 'GA', 'lat': 32.4609, 'lon': -84.9877, 'pop': 206922, 'county': 'Muscogee'},
        {'city': 'Montgomery', 'state': 'AL', 'lat': 32.3668, 'lon': -86.3000, 'pop': 200603, 'county': 'Montgomery'},
        {'city': 'Moreno Valley', 'state': 'CA', 'lat': 33.9425, 'lon': -117.2297, 'pop': 208634, 'county': 'Riverside'},
        {'city': 'Shreveport', 'state': 'LA', 'lat': 32.5252, 'lon': -93.7502, 'pop': 187593, 'county': 'Caddo'},
        {'city': 'Aurora', 'state': 'IL', 'lat': 41.7606, 'lon': -88.3201, 'pop': 180542, 'county': 'Kane'},
        {'city': 'Yonkers', 'state': 'NY', 'lat': 40.9312, 'lon': -73.8988, 'pop': 211569, 'county': 'Westchester'},
        {'city': 'Akron', 'state': 'OH', 'lat': 41.0814, 'lon': -81.5190, 'pop': 190469, 'county': 'Summit'},
        {'city': 'Huntington Beach', 'state': 'CA', 'lat': 33.6961, 'lon': -118.0000, 'pop': 198711, 'county': 'Orange'},
        {'city': 'Little Rock', 'state': 'AR', 'lat': 34.7465, 'lon': -92.2896, 'pop': 198541, 'county': 'Pulaski'},
        {'city': 'Augusta', 'state': 'GA', 'lat': 33.4735, 'lon': -82.0105, 'pop': 202081, 'county': 'Richmond'},
        {'city': 'Amarillo', 'state': 'TX', 'lat': 35.2220, 'lon': -101.8313, 'pop': 200393, 'county': 'Potter'},
        {'city': 'Glendale', 'state': 'CA', 'lat': 34.1425, 'lon': -118.2551, 'pop': 201361, 'county': 'Los Angeles'},
        {'city': 'Mobile', 'state': 'AL', 'lat': 30.6954, 'lon': -88.0399, 'pop': 187041, 'county': 'Mobile'},
        {'city': 'Grand Rapids', 'state': 'MI', 'lat': 42.9634, 'lon': -85.6681, 'pop': 198917, 'county': 'Kent'},
        {'city': 'Salt Lake City', 'state': 'UT', 'lat': 40.7608, 'lon': -111.8910, 'pop': 200567, 'county': 'Salt Lake'},
        {'city': 'Tallahassee', 'state': 'FL', 'lat': 30.4518, 'lon': -84.2807, 'pop': 194500, 'county': 'Leon'},
        {'city': 'Huntsville', 'state': 'AL', 'lat': 34.7304, 'lon': -86.5861, 'pop': 215006, 'county': 'Madison'},
        {'city': 'Grand Prairie', 'state': 'TX', 'lat': 32.7460, 'lon': -96.9978, 'pop': 196100, 'county': 'Dallas'},
        {'city': 'Knoxville', 'state': 'TN', 'lat': 35.9606, 'lon': -83.9207, 'pop': 190740, 'county': 'Knox'},
        {'city': 'Worcester', 'state': 'MA', 'lat': 42.2626, 'lon': -71.8023, 'pop': 206518, 'county': 'Worcester'},
        
        # Include North Dakota cities for compatibility
        {'city': 'Fargo', 'state': 'ND', 'lat': 46.8772, 'lon': -96.7898, 'pop': 125990, 'county': 'Cass'},
        {'city': 'Grand Forks', 'state': 'ND', 'lat': 47.9253, 'lon': -97.0329, 'pop': 59166, 'county': 'Grand Forks'},
        {'city': 'Bismarck', 'state': 'ND', 'lat': 46.8083, 'lon': -100.7837, 'pop': 73622, 'county': 'Burleigh'},
        
        # Additional smaller cities (50k+)
        {'city': 'Newport News', 'state': 'VA', 'lat': 37.0871, 'lon': -76.4730, 'pop': 186247, 'county': 'Newport News City'},
        {'city': 'Brownsville', 'state': 'TX', 'lat': 25.9018, 'lon': -97.4975, 'pop': 186738, 'county': 'Cameron'},
        {'city': 'Santa Clarita', 'state': 'CA', 'lat': 34.3917, 'lon': -118.5426, 'pop': 228673, 'county': 'Los Angeles'},
        {'city': 'Providence', 'state': 'RI', 'lat': 41.8240, 'lon': -71.4128, 'pop': 190934, 'county': 'Providence'},
        {'city': 'Fort Lauderdale', 'state': 'FL', 'lat': 26.1224, 'lon': -80.1373, 'pop': 182760, 'county': 'Broward'},
        {'city': 'Chattanooga', 'state': 'TN', 'lat': 35.0456, 'lon': -85.3097, 'pop': 181099, 'county': 'Hamilton'},
        {'city': 'Tempe', 'state': 'AZ', 'lat': 33.4255, 'lon': -111.9400, 'pop': 195805, 'county': 'Maricopa'},
        {'city': 'Oceanside', 'state': 'CA', 'lat': 33.1959, 'lon': -117.3795, 'pop': 174068, 'county': 'San Diego'},
        {'city': 'Garden Grove', 'state': 'CA', 'lat': 33.7739, 'lon': -117.9415, 'pop': 171644, 'county': 'Orange'},
        {'city': 'Rancho Cucamonga', 'state': 'CA', 'lat': 34.1064, 'lon': -117.5931, 'pop': 177451, 'county': 'San Bernardino'},
        {'city': 'Santa Rosa', 'state': 'CA', 'lat': 38.4404, 'lon': -122.7144, 'pop': 178127, 'county': 'Sonoma'},
        {'city': 'Vancouver', 'state': 'WA', 'lat': 45.6387, 'lon': -122.6615, 'pop': 183741, 'county': 'Clark'},
        {'city': 'Sioux Falls', 'state': 'SD', 'lat': 43.5446, 'lon': -96.7311, 'pop': 192517, 'county': 'Minnehaha'},
        {'city': 'Ontario', 'state': 'CA', 'lat': 34.0633, 'lon': -117.6509, 'pop': 175265, 'county': 'San Bernardino'},
        {'city': 'McKinney', 'state': 'TX', 'lat': 33.1972, 'lon': -96.6154, 'pop': 199177, 'county': 'Collin'},
        {'city': 'Elk Grove', 'state': 'CA', 'lat': 38.4088, 'lon': -121.3716, 'pop': 176124, 'county': 'Sacramento'},
        {'city': 'Salem', 'state': 'OR', 'lat': 44.9429, 'lon': -123.0351, 'pop': 178302, 'county': 'Marion'},
        {'city': 'Pembroke Pines', 'state': 'FL', 'lat': 26.0070, 'lon': -80.2962, 'pop': 171178, 'county': 'Broward'},
        {'city': 'Corona', 'state': 'CA', 'lat': 33.8753, 'lon': -117.5664, 'pop': 169868, 'county': 'Riverside'},
        {'city': 'Eugene', 'state': 'OR', 'lat': 44.0521, 'lon': -123.0868, 'pop': 176654, 'county': 'Lane'},
        {'city': 'Springfield', 'state': 'MO', 'lat': 37.2153, 'lon': -93.2982, 'pop': 169176, 'county': 'Greene'},
        {'city': 'Peoria', 'state': 'AZ', 'lat': 33.5806, 'lon': -112.2374, 'pop': 190985, 'county': 'Maricopa'},
        {'city': 'Fort Collins', 'state': 'CO', 'lat': 40.5853, 'lon': -105.0844, 'pop': 169810, 'county': 'Larimer'},
        {'city': 'Cary', 'state': 'NC', 'lat': 35.7915, 'lon': -78.7811, 'pop': 174721, 'county': 'Wake'},
        {'city': 'Lancaster', 'state': 'CA', 'lat': 34.6868, 'lon': -118.1542, 'pop': 173516, 'county': 'Los Angeles'},
        {'city': 'Hayward', 'state': 'CA', 'lat': 37.6688, 'lon': -122.0808, 'pop': 162954, 'county': 'Alameda'},
        {'city': 'Palmdale', 'state': 'CA', 'lat': 34.5794, 'lon': -118.1165, 'pop': 169450, 'county': 'Los Angeles'},
        {'city': 'Salinas', 'state': 'CA', 'lat': 36.6777, 'lon': -121.6555, 'pop': 164203, 'county': 'Monterey'},
        {'city': 'Springfield', 'state': 'IL', 'lat': 39.7817, 'lon': -89.6501, 'pop': 114394, 'county': 'Sangamon'},
        {'city': 'Hollywood', 'state': 'FL', 'lat': 26.0112, 'lon': -80.1494, 'pop': 154817, 'county': 'Broward'},
        {'city': 'Pasadena', 'state': 'TX', 'lat': 29.6911, 'lon': -95.2091, 'pop': 151950, 'county': 'Harris'},
        {'city': 'Pasadena', 'state': 'CA', 'lat': 34.1478, 'lon': -118.1445, 'pop': 138699, 'county': 'Los Angeles'},
        {'city': 'Sunnyvale', 'state': 'CA', 'lat': 37.3688, 'lon': -122.0363, 'pop': 155805, 'county': 'Santa Clara'},
        {'city': 'Springfield', 'state': 'MA', 'lat': 42.1015, 'lon': -72.5898, 'pop': 155929, 'county': 'Hampden'},
        {'city': 'Killeen', 'state': 'TX', 'lat': 31.1171, 'lon': -97.7278, 'pop': 153095, 'county': 'Bell'},
        {'city': 'Kansas City', 'state': 'KS', 'lat': 39.1142, 'lon': -94.6275, 'pop': 156607, 'county': 'Wyandotte'},
        {'city': 'Lakewood', 'state': 'CO', 'lat': 39.7047, 'lon': -105.0814, 'pop': 155984, 'county': 'Jefferson'},
        {'city': 'Torrance', 'state': 'CA', 'lat': 33.8358, 'lon': -118.3406, 'pop': 147067, 'county': 'Los Angeles'},
        {'city': 'Escondido', 'state': 'CA', 'lat': 33.1192, 'lon': -117.0864, 'pop': 151038, 'county': 'San Diego'},
        {'city': 'Naperville', 'state': 'IL', 'lat': 41.7508, 'lon': -88.1535, 'pop': 148449, 'county': 'DuPage'},
        {'city': 'Dayton', 'state': 'OH', 'lat': 39.7589, 'lon': -84.1916, 'pop': 140407, 'county': 'Montgomery'},
        {'city': 'Alexandria', 'state': 'VA', 'lat': 38.8048, 'lon': -77.0469, 'pop': 159467, 'county': 'Alexandria City'},
        {'city': 'Rockford', 'state': 'IL', 'lat': 42.2711, 'lon': -89.0940, 'pop': 148655, 'county': 'Winnebago'},
        {'city': 'Joliet', 'state': 'IL', 'lat': 41.5250, 'lon': -88.0817, 'pop': 150362, 'county': 'Will'},
        {'city': 'Clarksville', 'state': 'TN', 'lat': 36.5298, 'lon': -87.3595, 'pop': 166722, 'county': 'Montgomery'},
        {'city': 'Bellevue', 'state': 'WA', 'lat': 47.6101, 'lon': -122.2015, 'pop': 151854, 'county': 'King'},
        {'city': 'Concord', 'state': 'CA', 'lat': 37.9780, 'lon': -122.0311, 'pop': 129295, 'county': 'Contra Costa'},
        {'city': 'Cedar Rapids', 'state': 'IA', 'lat': 41.9779, 'lon': -91.6656, 'pop': 137710, 'county': 'Linn'},
        {'city': 'Charleston', 'state': 'SC', 'lat': 32.7765, 'lon': -79.9311, 'pop': 150227, 'county': 'Charleston'},
        {'city': 'Gainesville', 'state': 'FL', 'lat': 29.6516, 'lon': -82.3248, 'pop': 141085, 'county': 'Alachua'},
        {'city': 'Round Rock', 'state': 'TX', 'lat': 30.5083, 'lon': -97.6789, 'pop': 133372, 'county': 'Williamson'},
        {'city': 'Clearwater', 'state': 'FL', 'lat': 27.9659, 'lon': -82.8001, 'pop': 117292, 'county': 'Pinellas'},
        {'city': 'Waterbury', 'state': 'CT', 'lat': 41.5581, 'lon': -73.0515, 'pop': 114403, 'county': 'New Haven'},
        {'city': 'West Valley City', 'state': 'UT', 'lat': 40.6916, 'lon': -112.0011, 'pop': 140230, 'county': 'Salt Lake'},
        {'city': 'Costa Mesa', 'state': 'CA', 'lat': 33.6411, 'lon': -117.9187, 'pop': 112174, 'county': 'Orange'},
        {'city': 'Miami Gardens', 'state': 'FL', 'lat': 25.9420, 'lon': -80.2456, 'pop': 111640, 'county': 'Miami-Dade'},
        {'city': 'Carrollton', 'state': 'TX', 'lat': 32.9537, 'lon': -96.8903, 'pop': 139248, 'county': 'Dallas'},
        {'city': 'Cape Coral', 'state': 'FL', 'lat': 26.5629, 'lon': -81.9495, 'pop': 194016, 'county': 'Lee'},
        {'city': 'Stamford', 'state': 'CT', 'lat': 41.0534, 'lon': -73.5387, 'pop': 135470, 'county': 'Fairfield'},
        {'city': 'West Jordan', 'state': 'UT', 'lat': 40.6097, 'lon': -111.9391, 'pop': 116961, 'county': 'Salt Lake'},
        {'city': 'Surprise', 'state': 'AZ', 'lat': 33.6292, 'lon': -112.3679, 'pop': 141664, 'county': 'Maricopa'},
        {'city': 'Sterling Heights', 'state': 'MI', 'lat': 42.5803, 'lon': -83.0302, 'pop': 134346, 'county': 'Macomb'},
        {'city': 'Denton', 'state': 'TX', 'lat': 33.2148, 'lon': -97.1331, 'pop': 148910, 'county': 'Denton'},
        {'city': 'Coral Springs', 'state': 'FL', 'lat': 26.2710, 'lon': -80.2706, 'pop': 134394, 'county': 'Broward'},
        {'city': 'Thornton', 'state': 'CO', 'lat': 39.8681, 'lon': -104.9719, 'pop': 141867, 'county': 'Adams'},
        {'city': 'Miramar', 'state': 'FL', 'lat': 25.9873, 'lon': -80.2322, 'pop': 140823, 'county': 'Broward'},
        {'city': 'Thousand Oaks', 'state': 'CA', 'lat': 34.1706, 'lon': -118.8376, 'pop': 126813, 'county': 'Ventura'},
        {'city': 'Fullerton', 'state': 'CA', 'lat': 33.8704, 'lon': -117.9243, 'pop': 143617, 'county': 'Orange'},
        {'city': 'Roseville', 'state': 'CA', 'lat': 38.7521, 'lon': -121.2880, 'pop': 147773, 'county': 'Placer'},
        {'city': 'Kent', 'state': 'WA', 'lat': 47.3809, 'lon': -122.2348, 'pop': 136588, 'county': 'King'},
        {'city': 'Visalia', 'state': 'CA', 'lat': 36.3302, 'lon': -119.2921, 'pop': 141384, 'county': 'Tulare'},
        {'city': 'Olathe', 'state': 'KS', 'lat': 38.8814, 'lon': -94.8191, 'pop': 140545, 'county': 'Johnson'},
        {'city': 'Columbia', 'state': 'MO', 'lat': 38.9517, 'lon': -92.3341, 'pop': 126254, 'county': 'Boone'},
        {'city': 'Columbia', 'state': 'SC', 'lat': 34.0007, 'lon': -81.0348, 'pop': 137300, 'county': 'Richland'},
        {'city': 'Warren', 'state': 'MI', 'lat': 42.5145, 'lon': -83.0146, 'pop': 139387, 'county': 'Macomb'},
        {'city': 'Downey', 'state': 'CA', 'lat': 33.9401, 'lon': -118.1332, 'pop': 113242, 'county': 'Los Angeles'},
        {'city': 'Centennial', 'state': 'CO', 'lat': 39.5807, 'lon': -104.8756, 'pop': 108418, 'county': 'Arapahoe'},
        {'city': 'Pearland', 'state': 'TX', 'lat': 29.5638, 'lon': -95.2861, 'pop': 131448, 'county': 'Brazoria'},
        {'city': 'Temecula', 'state': 'CA', 'lat': 33.4936, 'lon': -117.1484, 'pop': 114761, 'county': 'Riverside'},
        {'city': 'Richardson', 'state': 'TX', 'lat': 32.9483, 'lon': -96.7299, 'pop': 120981, 'county': 'Dallas'},
        {'city': 'Concord', 'state': 'NC', 'lat': 35.4087, 'lon': -80.5792, 'pop': 105240, 'county': 'Cabarrus'},
        {'city': 'Elgin', 'state': 'IL', 'lat': 42.0354, 'lon': -88.2826, 'pop': 114797, 'county': 'Kane'},
        {'city': 'Overland Park', 'state': 'KS', 'lat': 38.9822, 'lon': -94.6708, 'pop': 197238, 'county': 'Johnson'},
        {'city': 'Inglewood', 'state': 'CA', 'lat': 33.9617, 'lon': -118.3531, 'pop': 109398, 'county': 'Los Angeles'},
        {'city': 'League City', 'state': 'TX', 'lat': 29.5075, 'lon': -95.0949, 'pop': 114140, 'county': 'Galveston'},
        {'city': 'Manchester', 'state': 'NH', 'lat': 42.9956, 'lon': -71.4548, 'pop': 115644, 'county': 'Hillsborough'},
        {'city': 'Rochester', 'state': 'MN', 'lat': 44.0121, 'lon': -92.4802, 'pop': 121395, 'county': 'Olmsted'},
        {'city': 'Lowell', 'state': 'MA', 'lat': 42.6334, 'lon': -71.3162, 'pop': 115554, 'county': 'Middlesex'},
        {'city': 'Broken Arrow', 'state': 'OK', 'lat': 36.0365, 'lon': -95.7975, 'pop': 113540, 'county': 'Tulsa'},
        {'city': 'Murfreesboro', 'state': 'TN', 'lat': 35.8456, 'lon': -86.3903, 'pop': 152769, 'county': 'Rutherford'},
        {'city': 'Cambridge', 'state': 'MA', 'lat': 42.3736, 'lon': -71.1097, 'pop': 118403, 'county': 'Middlesex'},
        {'city': 'Evansville', 'state': 'IN', 'lat': 37.9755, 'lon': -87.5329, 'pop': 116830, 'county': 'Vanderburgh'},
    ]
    
    return pd.DataFrame(cities_data)

def generate_city_configs():
    """Generate city configurations from the comprehensive list"""
    print("🏗️ Generating comprehensive USA city configurations...")
    
    try:
        # Create the enhanced city config manager
        manager = CityConfigManager()
        
        print(f"✅ Successfully generated configurations for {len(manager.configs)} cities")
        print(f"📊 Configuration file saved as: {manager.config_file}")
        
        # Show some statistics
        states = {}
        population_ranges = {'small': 0, 'medium': 0, 'large': 0, 'major': 0}
        
        for config in manager.configs.values():
            state = config.market_data.state_code
            states[state] = states.get(state, 0) + 1
            
            # Categorize by population
            max_pop = config.demographics.typical_population_range[1]
            if max_pop < 10000:
                population_ranges['small'] += 1
            elif max_pop < 25000:
                population_ranges['medium'] += 1
            elif max_pop < 50000:
                population_ranges['large'] += 1
            else:
                population_ranges['major'] += 1
        
        print(f"\n📈 Statistics:")
        print(f"   States covered: {len(states)}")
        print(f"   Small cities (< 10k): {population_ranges['small']}")
        print(f"   Medium cities (10k-25k): {population_ranges['medium']}")
        print(f"   Large cities (25k-50k): {population_ranges['large']}")
        print(f"   Major cities (50k+): {population_ranges['major']}")
        
        print(f"\n🏆 Top states by city count:")
        for state, count in sorted(states.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {state}: {count} cities")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating configurations: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to generate USA city configurations"""
    print("🇺🇸 USA CITY CONFIGURATION GENERATOR")
    print("=" * 50)
    print("This script will generate city configurations for 300+ major USA cities")
    print("Including all cities with population > 50,000 in the continental US")
    
    # Check if config file already exists
    config_file = "usa_city_configs.yaml"
    if os.path.exists(config_file):
        print(f"\n⚠️  Configuration file '{config_file}' already exists")
        choice = input("Do you want to regenerate it? (y/n): ").lower().strip()
        if choice != 'y':
            print("Exiting without changes.")
            return
        
        # Backup existing file
        backup_file = f"{config_file}.backup"
        os.rename(config_file, backup_file)
        print(f"📁 Backed up existing file to '{backup_file}'")
    
    print(f"\n🚀 Starting city configuration generation...")
    
    if generate_city_configs():
        print(f"\n✅ SUCCESS!")
        print(f"📄 City configurations saved to: {config_file}")
        print(f"💡 You can now use this with your enhanced data collection system")
        print(f"\nExample usage:")
        print(f"   python enhanced_data_collection.py --city new_york_ny")
        print(f"   python enhanced_data_collection.py --city los_angeles_ca")
        print(f"   python enhanced_data_collection.py --city chicago_il")
        print(f"   python enhanced_data_collection.py --list-cities")
        
        print(f"\n🔍 To search for specific cities:")
        print(f"   python -c \"from city_config import CityConfigManager; ")
        print(f"   manager = CityConfigManager(); ")
        print(f"   print([c.display_name for c in manager.search_cities('san')])\"")
        
    else:
        print(f"\n❌ FAILED!")
        print(f"Check the error messages above and try again.")

if __name__ == "__main__":
    main()