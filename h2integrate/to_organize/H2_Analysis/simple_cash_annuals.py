import numpy as np
import openmdao.jax as omj

def simple_cash_annuals(plant_useful_life_yr: float, equipment_useful_life_yr: int, capex, opex, amortization_interest):
    # Number of cycles for equipment
    # Normally I'd suggest using this for a rounding function. But we must produce an integer
    
    full_cycle = omj.round_down(plant_useful_life_yr/equipment_useful_life_yr)
    partial_cycle = plant_useful_life_yr - full_cycle

    # Payment period for equipment (assume payment period is life of equipment)



# -simple_cash_annuals(
#             useful_life,
#             useful_life, # this input 
#             electrolyzer_total_capital_cost,
#             electrolyzer_OM_cost,
#             0.03,
#         )


def simple_cash_annuals(
    plant_useful_life, equipment_useful_life, capex, opex, amortization_interest
):
    # Number of cycles for equipment
    # normally I'd recommend doing this slightly differently but full_cycle MUST be an integer
    # full_cycle = omj.round_down(plant_useful_life_yr/equipment_useful_life_yr)
    full_cycle = plant_useful_life // equipment_useful_life
    partial_cycle = plant_useful_life % equipment_useful_life

    # Payment period for equipment (assume payment period is life of equipment)
    N = full_cycle * equipment_useful_life
    af = capex * (
        (amortization_interest * (1 + amortization_interest) ** equipment_useful_life)
        / (((1 + amortization_interest) ** equipment_useful_life) - 1)
    )
    amortization = [af] * N

    # Payment period if plant's useful life is sorter than equipment
    # this will fail if equipment_useful_life is not an integer!
    # N = len(range(full_cycle * equipment_useful_life, plant_useful_life))
    
    print("range(full_cycle * equipment_useful_life, plant_useful_life)", range(full_cycle * equipment_useful_life, plant_useful_life))
    if N > 0:
        ap = capex * (
            (amortization_interest * (1 + amortization_interest) ** partial_cycle)
            / (((1 + amortization_interest) ** partial_cycle) - 1)
        )
        amortization += [ap] * N

    # Initialize annual cash flow array for useful life of plant
    opex_annuals = [opex] * plant_useful_life

    cash_flow_annuals = np.add(opex_annuals, amortization)
    # print(cash_flow_annuals)
    return cash_flow_annuals


if __name__ == "__main__":
    cash_flow_annuals = simple_cash_annuals(30, 8.5, 1000, 20, 0.03)
    print("cash_flow_annuals", cash_flow_annuals)
