import jax.numpy as jnp
import omj.jax as omj


def replacement_frequency(plant_useful_life: int, equipment_useful_life):
    # plant_useful_life in years
    # equipment_useful_life in hours
    # and equipment_useful_life must be in hours
    
    years_of_service = jnp.arange(1, plant_useful_life + 1)
    annual_online_hours = jnp.ones(plant_useful_life) * 8760 # [8760, 8760, 8760, 8760, 8760, 8760]
    cumulative_hours_online = annual_online_hours * years_of_service # [8760, 8760x2, 8760x3, 8760x4, 8760x5, 8760x6]
    cumulative_replacement_anually = cumulative_hours_online / equipment_useful_life # [0.3, 0.8, 1.3, 1.8, 2.3, 2.8]
    rounded_cumulative_replacement_anually = omj.smooth_floor(cumulative_replacement_anually) # [0, 0, 1, 1, 2, 2] 

    # identify the difference between each year
    difference = jnp.diff(rounded_cumulative_replacement_anually)

    # keep the total number of replacements in year 1 as-is
    replacements_per_year = jnp.concatenate([rounded_cumulative_replacement_anually[:1], difference])

    return replacements_per_year


if __name__ == "__main__":
    replacement_frequency(30, 65000)