"""Portfolio simulation examples."""

import datetime as dt

from openseries import (
    OpenFrame,
    OpenTimeSeries,
    ReturnSimulation,
    efficient_frontier,
    prepare_plot_data,
    sharpeplot,
)

if __name__ == "__main__":
    seed = 55
    simulated_weights = 5000
    frontier_points = 30

    simulations = ReturnSimulation.from_merton_jump_gbm(
        number_of_sims=4,
        trading_days=2512,
        mean_annual_return=0.05,
        mean_annual_vol=0.1,
        jumps_lamda=0.1,
        jumps_sigma=0.3,
        jumps_mu=-0.2,
        trading_days_in_year=252,
        seed=seed,
    )
    assets = OpenFrame(
        [
            OpenTimeSeries.from_df(
                simulations.to_dataframe(name="Asset", end=dt.date(2023, 12, 29)),
                column_nmbr=serie,
            )
            for serie in range(simulations.number_of_sims)
        ],
    ).to_cumret()

    current = OpenTimeSeries.from_df(
        assets.make_portfolio(
            name="Current Portfolio",
            weight_strat="eq_weights",
        ),
    )

    frontier, simulated, optimum = efficient_frontier(
        eframe=assets,
        num_ports=simulated_weights,
        seed=seed,
        frontier_points=frontier_points,
    )

    plotframe = prepare_plot_data(
        assets=assets,
        current=current,
        optimized=optimum,
    )

    _, _ = sharpeplot(
        sim_frame=simulated,
        line_frame=frontier,
        point_frame=plotframe,
        point_frame_mode="markers+text",
        title=False,
        add_logo=True,
        auto_open=True,
    )
