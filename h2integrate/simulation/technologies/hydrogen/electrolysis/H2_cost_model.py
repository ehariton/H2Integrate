import warnings
import openmdao.api as om
import jax.numpy as jnp
import openmdao.jax as omj 
from h2integrate.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals


Debug = True  # Set to true to see print statements in compute_primal


class basic_H2_cost_model(om.JaxExplicitComponent):
    """
    Basic cost modeling for a PEM electrolyzer.
    Looking at cost projections for PEM electrolyzers over years 2022, 2025, 2030, 2035.
    Electricity costs are calculated outside of hydrogen cost model

    Needs:
    Scaling factor for off-shore electrolysis
    Verifying numbers are appropriate for simplified cash flows
    Verify how H2 PTC works/factors into cash flows

    If offshore = True, then additional cost scaling is added to account for added difficulties for
    offshore installation, offshore=False means onshore
    """

    def initialize(self):
        self.options['default_shape'] = ()

        # Options get declared here
        self.options.declare('include_refurb_in_opex', types=(bool,), default=False)
        self.options.declare('offshore', types=(bool,), default=False)
        self.options.declare('program_record', types=(bool,), defualt=False)
        self.options.declare('capacity_based_OM', types=(bool,), default=True)

    def setup(self):
        self.options['use_jit'] = not (Debug)

        self.add_input('electrolyzer_capex_kw', units='kw')
        self.add_input('time_between_replacement', units='h')
        self.add_input('electrolyzer_size_mw', units='Mw')
        self.add_input('useful_life', units='year')
        self.add_input('atb_year', units='')
        self.add_input('electrical_generation_timeseries_kw', units='kw')
        self.add_input('hydrogen_annual_output', units='')
        self.add_input('PTC_USD_kg', units='kg')
        self.add_input('ITC_perc', units='')

        self.add_output('cf_h2_annuals', units='')
        self.add_output('electrolyzer_total_capital_cost', units='')
        self.add_output('electrolyzer_OM_cost', units='')
        self.add_output('h2_tax_credit', units='USD', size=10)
        self.add_output('h2_itc', size=30)
    
    # because our compute primal output depends on static options, in this case 
    # self.options['include_refurb_in_opex'], we must define a get_self_statics method. This method must
    # return a tuple of all static variables that affect the output of compute_primal. Their order
    # in the tuple doesn't matter.  If your component happens to have discrete inputs, do NOT return
    # them here. Discrete inputs would be passed into the compute_primal function individually, after
    # the continuous variables, but we don't have any discrete inputs in this example.
    def get_self_statics(self):
        # return value must be hashable.  Note that if we only had one static variable we would
        # still need to return a tuple containing that variable and so would need to follow the
        # variable name with a comma, for example: return (self.staticvar,)
        return (self.options['include_refurb_in_opex'], self.options['offshore'], self.options['program_record'])

    def compute_primal(self, electrolyzer_capex_kw, time_between_replacement, electrolyzer_size_mw, useful_life, atb_year,
        electrical_generation_timeseries_kw, hydrogen_annual_output, PTC_USD_kg, ITC_perc):

        # Basic information in our analysis
        kw_continuous = electrolyzer_size_mw * 1000

        # Capacity factor
        avg_generation = jnp.mean(electrical_generation_timeseries_kw)  # Avg Generation
        # print("avg_generation: ", avg_generation)
        
        # We handle if statements slightly differently in jax
        cap_factor = avg_generation / kw_continuous
        # if cap_factor > 1.0:
        #     cap_factor = 1.0
        #     warnings.warn(
        #         "Electrolyzer capacity factor would be greater than 1 with provided energy profile."
        #         " Capacity factor has been reduced to 1 for electrolyzer cost estimate purposes."
        #     )
        #                       condition      , if true,             if false      )
        cap_factor = jnp.where(cap_factor > 1.0, 1.0, cap_factor)
        

        # print(cap_factor)
        # if cap_factor != approx(1.0):
        #     raise(ValueError("Capacity factor must equal 1"))
        # print("cap_factor",cap_factor)

        # TODO: time_between_replacement and electrolyzer_capex_kw cannot be both and input and an output. 
        # you must establish what it is before it is input into this component.
        # 
        # #Apply PEM Cost Estimates based on year based on GPRA pathway (H2New)
        # if atb_year == 2022:
        #     electrolyzer_capex_kw = 1100     #[$/kW capacity] stack capital cost
        #     time_between_replacement = 40000    #[hrs]
        # elif atb_year == 2025:
        #     electrolyzer_capex_kw = 300
        #     time_between_replacement = 80000    #[hrs]
        # elif atb_year == 2030:
        #     electrolyzer_capex_kw = 150
        #     time_between_replacement = 80000    #[hrs]
        # elif atb_year == 2035:
        #     electrolyzer_capex_kw = 100
        #     time_between_replacement = 80000    #[hrs]

        # Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO Program Record)
        # https://www.hydrogen.energy.gov/pdfs/19009_h2_production_cost_pem_electrolysis_2019.pdf

        # Capital costs provide by Hydrogen Production Cost From PEM Electrolysis - 2019 (HFTO
        # Program Record)
        stack_capital_cost = 342  # [$/kW]
        mechanical_bop_cost = 36  # [$/kW] for a compressor
        electrical_bop_cost = 82  # [$/kW] for a rectifier

        # Installed capital cost
        stack_installation_factor = 12 / 100  # [%] for stack cost
        elec_installation_factor = 12 / 100  # [%] and electrical BOP

        # scale installation fraction if offshore (see Singlitico 2021 https://doi.org/10.1016/j.rset.2021.100005)
        stack_installation_factor *= 1 + offshore
        elec_installation_factor *= 1 + offshore

        # mechanical BOP install cost = 0%

        # Indirect capital cost as a percentage of installed capital cost
        site_prep = 2 / 100  # [%]
        engineering_design = 10 / 100  # [%]
        project_contingency = 15 / 100  # [%]
        permitting = 15 / 100  # [%]
        land = 250000  # [$]

        stack_replacment_cost = 15 / 100  # [% of installed capital cost]
        fixed_OM = 0.24  # [$/kg H2]

        # Chose to use numbers provided by GPRA pathways
        # if program_record:
        #     total_direct_electrolyzer_cost_kw = (
        #         (stack_capital_cost * (1 + stack_installation_factor))
        #         + mechanical_bop_cost
        #         + (electrical_bop_cost * (1 + elec_installation_factor))
        #     )
        # else:
        #     total_direct_electrolyzer_cost_kw = (
        #         (electrolyzer_capex_kw * (1 + stack_installation_factor))
        #         + mechanical_bop_cost
        #         + (electrical_bop_cost * (1 + elec_installation_factor))
        #     )
        temp1 = (
                (stack_capital_cost * (1 + stack_installation_factor))
                + mechanical_bop_cost
                + (electrical_bop_cost * (1 + elec_installation_factor))
            )
        temp2  = (
                (electrolyzer_capex_kw * (1 + stack_installation_factor))
                + mechanical_bop_cost
                + (electrical_bop_cost * (1 + elec_installation_factor))
            )
        total_direct_electrolyzer_cost_kw = jnp.where(self.options['program_record'], temp1, temp2)


        # Assign CapEx for electrolyzer from capacity based installed CapEx
        electrolyzer_total_installed_capex = (
            total_direct_electrolyzer_cost_kw * electrolyzer_size_mw * 1000
        )

        # Add indirect capital costs
        electrolyzer_total_capital_cost = (
            (
                (site_prep + engineering_design + project_contingency + permitting)
                * electrolyzer_total_installed_capex
            )
            + land
            + electrolyzer_total_installed_capex
        )

        # O&M costs
        # https://www.sciencedirect.com/science/article/pii/S2542435121003068
        # for 700 MW electrolyzer (https://www.hydrogen.energy.gov/pdfs/19009_h2_production_cost_pem_electrolysis_2019.pdf)
        h2_FOM_kg = 0.24  # [$/kg]

        # linearly scaled current central fixed O&M for a 700MW electrolyzer up to a 1000MW electrolyzer
        scaled_h2_FOM_kg = h2_FOM_kg * electrolyzer_size_mw / 700

        h2_FOM_kWh = scaled_h2_FOM_kg / 55.5  # [$/kWh] used 55.5 kWh/kg for efficiency
        fixed_OM = h2_FOM_kWh * 8760  # [$/kW-y]
        property_tax_insurance = 1.5 / 100  # [% of Cap/y]
        variable_OM = 1.30  # [$/MWh]

        # Amortized refurbishment expense [$/MWh]
        # if not include_refurb_in_opex:
        #     amortized_refurbish_cost = 0.0
        # else:
        #     amortized_refurbish_cost = (
        #         (total_direct_electrolyzer_cost_kw * stack_replacment_cost)
        #         * max(((useful_life * 8760 * cap_factor) / time_between_replacement - 1), 0)
        #         / useful_life
        #         / 8760
        #         / cap_factor
        #         * 1000
        #     )
        amortized_refurbish_cost = (
                (total_direct_electrolyzer_cost_kw * stack_replacment_cost)
                * max(((useful_life * 8760 * cap_factor) / time_between_replacement - 1), 0)
                / useful_life
                / 8760
                / cap_factor
                * 1000
            )
        amortized_refurbish_cost = jnp.where(self.options['include_refurb_in_opex'], amortized_refurbish_cost, 0.0)
        

        # Total O&M costs [% of installed cap/year]
        total_OM_costs = (
            fixed_OM + (property_tax_insurance * total_direct_electrolyzer_cost_kw)
        ) / total_direct_electrolyzer_cost_kw + (
            (variable_OM + amortized_refurbish_cost)
            / 1000
            * 8760
            * (cap_factor / total_direct_electrolyzer_cost_kw)
        )

        # capacity_based_OM = True
        # if capacity_based_OM:
        #     electrolyzer_OM_cost = electrolyzer_total_installed_capex * total_OM_costs  # Capacity based
        # else:
        #     electrolyzer_OM_cost = (
        #         fixed_OM * hydrogen_annual_output
        #     )  # Production based - likely not very accurate
        electrolyzer_OM_cost = jnp.where(self.options['capacity_based_OM'], electrolyzer_total_installed_capex * total_OM_costs, fixed_OM * hydrogen_annual_output)

        # Add in electrolyzer repair schedule (every 7 years)
        # Use if not using time between replacement given in hours
        # Currently not added into further calculations

        # TODO: The original calculations do not have the correct units because useful_life 
        # is meansured in years and time_between_replacement is measured in hours.
        # We are going to skip building the repair schedule since it should probably not be part
        # of the cost calculations. You could probably solve this with a lax.fori_loop(). 
        
        # electrolyzer_repair_schedule = jnp.array[]
        # counter = 1
        # for year in range(0, useful_life):
        #     if year == 0:
        #         electrolyzer_repair_schedule = np.append(electrolyzer_repair_schedule, [0])

        #     elif counter % time_between_replacement == 0:
        #         electrolyzer_repair_schedule = np.append(electrolyzer_repair_schedule, [1])

        #     else:
        #         electrolyzer_repair_schedule = np.append(electrolyzer_repair_schedule, [0])
        #     counter += 1
        # electrolyzer_repair_schedule * (stack_replacment_cost * electrolyzer_total_installed_capex)
        # print("H2 replacement costs: ", electrolyzer_replacement_costs)

        # TMP Thinking area:
        # def body(i, arr):
        #     return arr.at[i].set(i)

        # # an array of integer years
        # result = lax.fori_loop(0, useful_life, body, jnp.zeros(useful_life))
        # # Output: [0. 1. 2. 3. 4.]

        # # How many hours has the electrolizer run
        # electro_hours = result * 8760

        # # run hours divided by replacement hours
        # electro_hours_per_replacement = electro_hours / electrolyzer_repair_schedule

        # # number of replacements
        # omj.smooth_round(electro_hours_per_replacement)
        # # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, ...]

        # electrolyzer_repair_schedule = jnp.zeros(useful_life) # Pre-allocate an array of zeros

        # #                                          init value 
        # tmp_useful_life = useful_life
        # result = lax.fori_loop(0, useful_life, body, jnp.zeros(useful_life))

        # from jax import lax

        # def body(i, electrolyzer_repair_schedule):
        #     electrolyzer_repair_schedule = jnp.where(tmp_useful_life >= time_between_replacement / 8766, 1, 0)
        #     tmp_useful_life = jnp.where(tmp_useful_life >= time_between_replacement / 8766, tmp_useful_life - time_between_replacement / 8766, tmp_useful_life)
            
        #     return electrolyzer_repair_schedule.at[i].set(i)




        # Include Hydrogen PTC from the Inflation Reduction Act (range $0.60 - $3/kg-H2)
        h2_tax_credit = jnp.ones(10) * (hydrogen_annual_output * PTC_USD_kg)
        # print('H2 tax credit',h2_tax_credit)

        # Include ITC from IRA (range 0% - 50%)
        # ITC is expressed as a percentage of the total installed cost which reduces the annual tax
        # liabiity in year one of the project cash flow.
        # h2_itc = (ITC_perc / 100) * electrolyzer_total_installed_capex
        # cf_h2_itc = [0] * 30
        # cf_h2_itc[1] = h2_itc

        cf_h2_itc = jnp.zeros(30)
        cf_h2_itc = cf_h2_itc.at[1].set((ITC_perc / 100) * electrolyzer_total_installed_capex)
        # print('ITC', cf_h2_itc)

        # Simple cash annuals
        cf_h2_annuals = -simple_cash_annuals(
            useful_life,
            useful_life,
            electrolyzer_total_capital_cost,
            electrolyzer_OM_cost,
            0.03,
        )

        # print("CF H2 Annuals",cf_h2_annuals)

        # Add positive cashflow from tax credit
        cf_h2_annuals = jnp.add(cf_h2_annuals, h2_tax_credit)

        # print('Added H2 ptc with cash flows', cf_h2_annuals)

        # Add ITC
        cf_h2_annuals = jnp.add(cf_h2_itc, cf_h2_annuals)
        # print('Added H2 ITC with cash flows', cf_h2_annuals)

        return (
            cf_h2_annuals,
            electrolyzer_total_capital_cost,
            electrolyzer_OM_cost,
            electrolyzer_capex_kw,
            # time_between_replacement,
            h2_tax_credit,
            h2_itc,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # plot a sweep of sizes for OPEX and CAPEX

    electrolyzer_capex_kw = 1300  # $/kW
    time_between_replacement = 65000  # hours
    electrolyzer_sizes_mw = np.arange(1, 1000)
    useful_life = 30  # years
    atb_year = 2025
    # electrical_generation_timeseries_kw = np.sin(np.arange(0,24*365)*1E-3)*0.5E6 + 0.6E6
    hydrogen_annual_output = 0

    # for distributed
    ndivs = [2, 5, 10]

    opex = []
    capex = []
    opex_distributed = np.zeros((len(ndivs), len(electrolyzer_sizes_mw)))
    capex_distributed = np.zeros((len(ndivs), len(electrolyzer_sizes_mw)))

    for i, electrolyzer_size_mw in enumerate(electrolyzer_sizes_mw):
        electrical_generation_timeseries_kw = electrolyzer_size_mw * 1000 * np.ones(365 * 24)

        # centralized
        _, electrolyzer_total_capital_cost, electrolyzer_OM_cost, _, _, _, _ = basic_H2_cost_model(
            electrolyzer_capex_kw,
            time_between_replacement,
            electrolyzer_size_mw,
            useful_life,
            atb_year,
            electrical_generation_timeseries_kw,
            hydrogen_annual_output,
            0,
            0,
            include_refurb_in_opex=False,
            offshore=0,
        )

        opex.append(electrolyzer_OM_cost)
        capex.append(electrolyzer_total_capital_cost)

        for j, div in enumerate(ndivs):
            # divided
            electrolyzer_size_mw_distributed = electrolyzer_size_mw / div
            electrical_generation_timeseries_kw_distibuted = (
                electrical_generation_timeseries_kw / div
            )

            (
                _,
                electrolyzer_capital_cost_distributed,
                electrolyzer_OM_cost_distributed,
                electrolyzer_capex_kw_distributed,
                time_between_replacement,
                h2_tax_credit,
                h2_itc,
            ) = basic_H2_cost_model(
                electrolyzer_capex_kw,
                time_between_replacement,
                electrolyzer_size_mw_distributed,
                useful_life,
                atb_year,
                electrical_generation_timeseries_kw_distibuted,
                hydrogen_annual_output,
                0,
                0,
                include_refurb_in_opex=False,
                offshore=0,
            )
            # print(opex_distributed)
            opex_distributed[j, i] = electrolyzer_OM_cost_distributed * div
            capex_distributed[j, i] = electrolyzer_capital_cost_distributed * div

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].plot(electrolyzer_sizes_mw, np.asarray(capex) * 1e-6, label="Centralized")
    ax[1].plot(electrolyzer_sizes_mw, np.asarray(opex) * 1e-6, label="Centralized")

    for i, div in enumerate(ndivs):
        # dims(capex_distributed)
        ax[0].plot(
            electrolyzer_sizes_mw,
            np.asarray(capex_distributed[i]) * 1e-6,
            "--",
            label=f"{div} Divisions",
        )
        ax[1].plot(
            electrolyzer_sizes_mw,
            np.asarray(opex_distributed[i]) * 1e-6,
            "--",
            label=f"{div} Divisions",
        )

    ax[0].set(ylabel="CAPEX (M USD)", xlabel="Electrolyzer Size (MW)")
    ax[1].set(ylabel="Annual OPEX (M USD)", xlabel="Electrolyzer Size (MW)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    ## plot divided energy signals
    fig, ax = plt.subplots(1)
    ax.plot(electrical_generation_timeseries_kw, label="1")
    for div in ndivs:
        ax.plot(electrical_generation_timeseries_kw / div, label=f"{div}")

    ax.set(xlabel="Hour", ylabel="Power (MW)")
    plt.tight_layout()
    plt.show()
