

#The true demand of crop.
def True_Demand(weather_s, grow_stage):
    stage_factor = [0.5, 0.7, 1.0, 1.2, 1.0][grow_stage]
    weather_factor = {"Drought": 1.2, "Normal": 1.0, "Rainy": 0.8}[weather_s]
    base_demand = 10
    return base_demand * stage_factor * weather_factor