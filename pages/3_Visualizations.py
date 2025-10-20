import streamlit as st
import charts
from NotebookCreator import create_notebook
from streamlit_extras.switch_page_button import switch_page

from nbconvert import PythonExporter
import nbformat as nbf
from CompositionFunctions import *
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(layout="wide")
st.session_state['active_page'] = 'Visualization_page'

st.title('Visualization Playground')

if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = None
df = st.session_state['selected_dataset']

if 'hide_non_feasible_values' not in st.session_state:
    st.session_state['hide_non_feasible_values'] = False

if st.session_state['selected_dataset'] is None:
    st.session_state['selected_dataset'] = pd.read_csv('https://raw.githubusercontent.com/lpanavas/DPEducationDatasets/master/PUMS_california_demographics_1000.csv')
    df = st.session_state['selected_dataset']
    st.session_state['dataset_url'] = 'https://raw.githubusercontent.com/lpanavas/DPEducationDatasets/master/PUMS_california_demographics_1000.csv'


# Tab 1
def render_simulations(df):
    st.header("Simulations")
    st.write("Here we will be able to see simulated private outputs on each of your queries. Choose the query you want and select variations of different parameters to explore.")
    col1, col2 = st.columns([2, 3], gap="large")

    # Initialize an empty list to store the numbers
    if 'simulations_parameter_selection' not in st.session_state:
        st.session_state['simulations_parameter_selection'] = None

    with col1:
        st.header("Experimental Parameters")
        st.write("Please select parameters.")
        # Initialize session state for query info if not exists
        if 'query_info' not in st.session_state:
            st.session_state['query_info'] = {
                'column': None,
                'query_type': 'Average',
                'data_type': None,
                'epsilon': 1.000,
                'average_bounds': {'lower_bound': None, 'upper_bound': None},
                'histogram_bounds': {'lower_bound': None, 'upper_bound': None, 'bins': 10},
            }

        def update_query_info(key, value):
            st.session_state['query_info'][key] = value

        if df is not None:
            column_names = df.columns.tolist()
            left_col1, left_col2 = st.columns(2)

            with left_col1:
                selected_column = st.selectbox(
                    "Select a column for analysis",
                    column_names,
                    key='selected_column',
                    on_change=lambda: update_query_info('column', st.session_state['selected_column'])
                )
                st.session_state['query_info']['column'] = selected_column

            # Determine if column is categorical
            unique_values = df[selected_column].nunique()
            is_categorical = unique_values <= 20
            st.session_state['query_info']['data_type'] = "categorical" if is_categorical else "continuous"

            # Set query type options based on data type
            options = ["Count", "Histogram"] if is_categorical else ["Count", "Average", "Histogram"]
            with left_col2:
                analysis_option = st.radio(
                    "Choose an analysis option:",
                    options,
                    index=options.index('Average') if 'Average' in options else 0,
                    key='analysis_option',
                    on_change=lambda: update_query_info('query_type', st.session_state['analysis_option'])
                )
                st.session_state['query_info']['query_type'] = analysis_option

            def update_query_info_bounds(key, subkey, value):
                st.session_state['query_info'][key][subkey] = value

            # Add bounds input for continuous variables
            if not is_categorical:
                if analysis_option == "Average":
                    st.subheader("Input Bounds for Average")
                    lower_col, upper_col = st.columns(2)
                    with lower_col:
                        st.number_input(
                            "Lower Bound",
                            step=1.0,
                            value=st.session_state['query_info'].get('average_bounds', {}).get('lower_bound', None),
                            key="average_lower_bound",
                            on_change=lambda: update_query_info_bounds("average_bounds", "lower_bound", st.session_state["average_lower_bound"]),
                        )
                    with upper_col:
                        st.number_input(
                            "Upper Bound",
                            step=1.0,
                            value=st.session_state['query_info'].get('average_bounds', {}).get('upper_bound', None),
                            key="average_upper_bound",
                            on_change=lambda: update_query_info_bounds("average_bounds", "upper_bound", st.session_state["average_upper_bound"]),
                        )
                elif analysis_option == "Histogram":
                    st.subheader("Histogram Options")
                    lower_col, upper_col = st.columns(2)
                    with lower_col:
                        st.number_input(
                            "Lower Bound",
                            step=1.0,
                            value=st.session_state['query_info']['histogram_bounds']['lower_bound'],
                            key="hist_lower_bound",
                            on_change=lambda: update_query_info_bounds("histogram_bounds", "lower_bound", st.session_state["hist_lower_bound"]),
                        )
                    with upper_col:
                        st.number_input(
                            "Upper Bound",
                            step=1.0,
                            value=st.session_state['query_info']['histogram_bounds']['upper_bound'],
                            key="hist_upper_bound",
                            on_change=lambda: update_query_info_bounds("histogram_bounds", "upper_bound", st.session_state["hist_upper_bound"]),
                        )
                    st.number_input(
                        "Number of Bins",
                        min_value=2,
                        max_value=50,
                        value=st.session_state['query_info']['histogram_bounds']['bins'],
                        key='hist_bins',
                        on_change=lambda: update_query_info_bounds("histogram_bounds", "bins", st.session_state["hist_bins"]),
                    )

            # Set selected_query for compatibility with existing code
            if st.session_state['query_info']['column'] and st.session_state['query_info']['query_type']:
                selected_query = f"{st.session_state['query_info']['column']}_{st.session_state['query_info']['query_type'].lower()}"
                st.session_state['visualize_clicked'] = False
            else:
                selected_query = None

            # Maintain parameter selection for compatibility
            options = ['Epsilon', 'Mechanism']
            # Only show parameter selection if bounds are set for Average or Histogram
            show_parameter_selection = True
            if not is_categorical:
                if analysis_option == "Average":
                    if (st.session_state['query_info']['average_bounds']['lower_bound'] is None or 
                        st.session_state['query_info']['average_bounds']['upper_bound'] is None):
                        show_parameter_selection = False
                    else:
                        options.append('Bounds')
                elif analysis_option == "Histogram":
                    if (st.session_state['query_info']['histogram_bounds']['lower_bound'] is None or 
                        st.session_state['query_info']['histogram_bounds']['upper_bound'] is None):
                        show_parameter_selection = False
                    else:
                        options.append('Bins')

            # Only show parameter selection if conditions are met
            if show_parameter_selection:
                # Determine the index before passing it to selectbox
                if st.session_state['simulations_parameter_selection'] in options:
                    selected_index = options.index(st.session_state['simulations_parameter_selection'])
                else:
                    selected_index = None
                    st.session_state['simulations_parameter_selection'] = None

                selected_option = st.selectbox(
                    "Which implementation parameter are you interested in?",
                    options,
                    index=selected_index,
                    placeholder="Select Parameter ...",
                )
                if selected_option and selected_option != st.session_state['simulations_parameter_selection']:
                    st.session_state['simulations_parameter_selection'] = selected_option

                if st.session_state['simulations_parameter_selection']:
                    if st.session_state['simulations_parameter_selection'] == 'Epsilon':
                        if 'epsilon_inputs' not in st.session_state:
                            st.session_state['epsilon_inputs'] = [1.000]  # Add the default epsilon value here
                        col3, col4 = st.columns([1, 1])
                        with col3:
                            with st.form('epsilon_form'):
                                st.write('Enter \u03B5 Values (max 4):')
                                new_epsilon = st.number_input('Privacy (\u03B5)', min_value=0.001, step=0.001, value=1.0000, format="%.3f", key='epsilon_input')
                                submitted = st.form_submit_button('Add \u03B5 value')
                                if submitted:
                                    if len(st.session_state['epsilon_inputs']) < 4:
                                        if new_epsilon not in st.session_state['epsilon_inputs']:
                                            st.session_state['epsilon_inputs'].append(new_epsilon)
                                        else:
                                            st.warning('This \u03B5 value has already been added.')
                                    else: 
                                        st.warning('Maximum of 4 \u03B5 values reached.')
                        with col4:
                            for index, epsilon in enumerate(st.session_state['epsilon_inputs']):
                                if st.button(f"Delete {epsilon}", key=f"delete_{index}"):
                                    st.session_state['epsilon_inputs'].remove(epsilon)
                                    st.rerun()

                    elif st.session_state['simulations_parameter_selection'] == 'Bounds':
                        epsilon = st.number_input('Epsilon', min_value=0.001, step=0.001, value=1.0000, format="%.3f", key='epsilon_input')
                        col3, col4 = st.columns([1, 1])
                        with col3:
                            st.write('Enter Lower and Upper Bounds:')
                            lower_bound = st.number_input('Lower Bound', min_value=-100000000.0, step=0.1, value=st.session_state['query_info']['average_bounds']['lower_bound'], format="%.1f", key='lower_bound_input')
                            upper_bound = st.number_input('Upper Bound', min_value=lower_bound, step=0.1, value=st.session_state['query_info']['average_bounds']['upper_bound'], format="%.1f", key='upper_bound_input')
                            if 'bounds_inputs' not in st.session_state:
                                st.session_state['bounds_inputs'] = [(lower_bound, upper_bound)]
                            submitted = st.button('Add Bounds')
                            if submitted:
                                new_bounds = (lower_bound, upper_bound)
                                if len(st.session_state['bounds_inputs']) < 4:
                                    if new_bounds not in st.session_state['bounds_inputs']:
                                        st.session_state['bounds_inputs'].append(new_bounds)
                                    else:
                                        st.warning('This Bounds value has already been added.')
                                else: 
                                    st.warning('Maximum of 4 Bounds values reached.')
                        with col4:
                            for index, bounds in enumerate(st.session_state['bounds_inputs']):
                                if st.button(f"Delete {bounds}", key=f"delete_{index}"):
                                    st.session_state['bounds_inputs'].remove(bounds)
                                    st.rerun()

                    elif st.session_state['simulations_parameter_selection'] == 'Bins':
                        epsilon = st.number_input('Epsilon', min_value=0.001, step=0.001, value=1.0000, format="%.3f", key='epsilon_input')
                        col3, col4 = st.columns([1, 1])
                        with col3:
                            lower_bound = st.session_state['query_info']['histogram_bounds']['lower_bound']
                            num_bins = st.number_input('Number of Bins', min_value=2, max_value=50, value=10, step=1, key='num_bins_input')
                        with col3:
                            if 'bins_inputs' not in st.session_state:
                                st.session_state['bins_inputs'] = [num_bins]
                            submitted = st.button('Add Bins')
                            if submitted:
                                new_bins = (num_bins)
                                if len(st.session_state['bins_inputs']) < 4:
                                    if new_bins not in st.session_state['bins_inputs']:
                                        st.session_state['bins_inputs'].append(new_bins)
                                    else:
                                        st.warning('This Bins value has already been added.')
                                else: 
                                    st.warning('Maximum of 4 Bins values reached.')
                            with col4:       
                                upper_bound = st.session_state['query_info']['histogram_bounds']['upper_bound']
                                st.write('Selected Bins')
                                for index, bins in enumerate(st.session_state['bins_inputs']):
                                    if st.button(f"Delete {bins}", key=f"delete_{index}"):
                                        st.session_state['bins_inputs'].remove(bins)
                                        st.rerun()

                    elif st.session_state['simulations_parameter_selection'] == 'Mechanism':
                        epsilon = st.number_input('Epsilon', min_value=0.001, step=0.001, value=1.0000, format="%.3f", key='epsilon_input')
                        mechanisms = st.multiselect(
                            'Which mechanism do you want displayed',
                            ['gaussian', 'laplace'],
                            ['gaussian', 'laplace'],
                            key='simulations_selected_mechanism'
                        )

                    if st.session_state['query_info']['column'] and st.session_state['query_info']['query_type']:
                        columnName, queryType = selected_query.split('_')
                        if queryType in ['count', 'histogram']:
                            hide_non_feasible_values = st.checkbox(
                                "Hide non-feasible values", value=False, key='hide_non_feasible_values',
                                help="Because of the added noise, counts can occasionally become negative. Post-processing typically removes such values. Check this to filter values < 0."
                            )

    with col2:
        st.header("Visualization")
        st.markdown("When you have selected your parameters, **click the Visualize button** to see them. *Note: If you change any parameters, you will need to click the Visualize button again to update the visualization.*", unsafe_allow_html=True)

        if 'visualize_clicked' not in st.session_state or st.button("Visualize"):
            st.session_state['visualize_clicked'] = True
            if st.session_state['simulations_parameter_selection'] == 'Epsilon':
                if st.session_state['query_info']['query_type'] in ['Average', 'Count']:
                    chart_config = charts.preset_parameters(df, selected_query, st.session_state['simulations_parameter_selection'], st.session_state['epsilon_inputs'], hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                    st.session_state['fig'] = chart_config
                else:
                    if st.session_state['query_info']['data_type'] == 'categorical':
                        chart_config = charts.noisy_histogram_creation(df, selected_query, st.session_state['query_info']['data_type'], st.session_state['simulations_parameter_selection'], st.session_state['epsilon_inputs'], st.session_state['query_info']['histogram_bounds']['lower_bound'], st.session_state['query_info']['histogram_bounds']['upper_bound'], st.session_state['query_info']['histogram_bounds']['bins'], epsilon_input=None, hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                        st.session_state['fig'] = chart_config
                    else:
                        chart_config = charts.noisy_histogram_creation(df, selected_query, st.session_state['query_info']['data_type'], st.session_state['simulations_parameter_selection'], st.session_state['epsilon_inputs'], st.session_state['query_info']['histogram_bounds']['lower_bound'], st.session_state['query_info']['histogram_bounds']['upper_bound'], st.session_state['query_info']['histogram_bounds']['bins'], epsilon_input=None, hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                        st.session_state['fig'] = chart_config

            if st.session_state['simulations_parameter_selection'] == 'Bounds':
                chart_config = charts.preset_parameters(df, selected_query, st.session_state['simulations_parameter_selection'], st.session_state['bounds_inputs'], hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                st.session_state['fig'] = chart_config

            if st.session_state['simulations_parameter_selection'] == 'Bins':
                chart_config = charts.noisy_histogram_creation(df, selected_query, st.session_state['query_info']['data_type'], st.session_state['simulations_parameter_selection'], st.session_state['bins_inputs'], st.session_state['query_info']['histogram_bounds']['lower_bound'], st.session_state['query_info']['histogram_bounds']['upper_bound'], st.session_state['query_info']['histogram_bounds']['bins'], st.session_state['epsilon_input'], hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                st.session_state['fig'] = chart_config

            if st.session_state['simulations_parameter_selection'] == 'Mechanism':
                if st.session_state['query_info']['query_type'] in ['Average', 'Count']:
                    chart_config = charts.preset_parameters(df, selected_query, st.session_state['simulations_parameter_selection'], st.session_state['simulations_selected_mechanism'], hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                    st.session_state['fig'] = chart_config
                else:
                    if st.session_state['query_info']['data_type'] == 'categorical':
                        chart_config = charts.noisy_histogram_creation(df, selected_query, st.session_state['query_info']['data_type'], st.session_state['simulations_parameter_selection'], st.session_state['simulations_selected_mechanism'], st.session_state['query_info']['histogram_bounds']['lower_bound'], st.session_state['query_info']['histogram_bounds']['upper_bound'], st.session_state['query_info']['histogram_bounds']['bins'], st.session_state['epsilon_input'], hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                        st.session_state['fig'] = chart_config
                    else:
                        chart_config = charts.noisy_histogram_creation(df, selected_query, st.session_state['query_info']['data_type'], st.session_state['simulations_parameter_selection'], st.session_state['simulations_selected_mechanism'], st.session_state['query_info']['histogram_bounds']['lower_bound'], st.session_state['query_info']['histogram_bounds']['upper_bound'], st.session_state['query_info']['histogram_bounds']['bins'], st.session_state['epsilon_input'], hide_non_feasible_values=st.session_state['hide_non_feasible_values'])
                        st.session_state['fig'] = chart_config

        if 'visualize_clicked' in st.session_state and st.session_state['visualize_clicked']:
            fig = st.session_state.get('fig', None)
            if fig is not None:
                with st.spinner('Loading Visualization...'):
                    st.plotly_chart(fig, use_container_width=True)


def render_one_query(df):
    # Corresponds to the original tab2
    col1, col2 = st.columns([1, 2], gap="large")

    if 'one_query_selected_mechanism' not in st.session_state:
        st.session_state['one_query_selected_mechanism'] = None
    if 'alpha' not in st.session_state:
        st.session_state['alpha'] = None
    if 'error_type' not in st.session_state:
        st.session_state['error_type'] = None
    if 'one_query_info' not in st.session_state:
        st.session_state['one_query_info'] = {
            'column': None,
            'query_type': 'Average',
            'data_type': None,
            'average_bounds': {'lower_bound': None, 'upper_bound': None},
            'histogram_bounds': {'lower_bound': None, 'upper_bound': None, 'bins': 10},
        }

    def update_one_query_info(key, value):
        st.session_state['one_query_info'][key] = value

    def update_one_query_info_bounds(key, subkey, value):
        st.session_state['one_query_info'][key][subkey] = value

    with col1:
        st.header('Inputs')
        if df is not None:
            column_names = df.columns.tolist()
            left_col1, left_col2 = st.columns(2)

            with left_col1:
                selected_column = st.selectbox(
                    "Select a column for analysis",
                    column_names,
                    key='one_query_selected_column',
                    on_change=lambda: update_one_query_info('column', st.session_state['one_query_selected_column'])
                )
                st.session_state['one_query_info']['column'] = selected_column

                unique_values = df[selected_column].nunique()
                is_categorical = unique_values <= 20
                st.session_state['one_query_info']['data_type'] = "categorical" if is_categorical else "continuous"

            options = ["Count", "Histogram"] if is_categorical else ["Count", "Average", "Histogram"]
            with left_col2:
                analysis_option = st.radio(
                    "Choose an analysis option:",
                    options,
                    index=options.index('Average') if 'Average' in options else 0,
                    key='one_query_analysis_option',
                    on_change=lambda: update_one_query_info('query_type', st.session_state['one_query_analysis_option'])
                )
                st.session_state['one_query_info']['query_type'] = analysis_option

            if not is_categorical:
                if analysis_option == "Average":
                    st.subheader("Input Bounds for Average")
                    lower_col, upper_col = st.columns(2)
                    with lower_col:
                        st.number_input(
                            "Lower Bound",
                            step=1.0,
                            value=st.session_state['one_query_info']['average_bounds']['lower_bound'],
                            key="one_query_average_lower_bound",
                            on_change=lambda: update_one_query_info_bounds("average_bounds", "lower_bound", st.session_state["one_query_average_lower_bound"]),
                        )
                    with upper_col:
                        st.number_input(
                            "Upper Bound",
                            step=1.0,
                            value=st.session_state['one_query_info']['average_bounds']['upper_bound'],
                            key="one_query_average_upper_bound",
                            on_change=lambda: update_one_query_info_bounds("average_bounds", "upper_bound", st.session_state["one_query_average_upper_bound"]),
                        )
                elif analysis_option == "Histogram":
                    st.subheader("Histogram Options")
                    lower_col, upper_col = st.columns(2)
                    with lower_col:
                        st.number_input(
                            "Lower Bound",
                            step=1.0,
                            value=st.session_state['one_query_info']['histogram_bounds']['lower_bound'],
                            key="one_query_hist_lower_bound",
                            on_change=lambda: update_one_query_info_bounds("histogram_bounds", "lower_bound", st.session_state["one_query_hist_lower_bound"]),
                        )
                    with upper_col:
                        st.number_input(
                            "Upper Bound",
                            step=1.0,
                            value=st.session_state['one_query_info']['histogram_bounds']['upper_bound'],
                            key="one_query_hist_upper_bound",
                            on_change=lambda: update_one_query_info_bounds("histogram_bounds", "upper_bound", st.session_state["one_query_hist_upper_bound"]),
                        )
                    st.number_input(
                        "Histogram Bins",
                        step=1,
                        value=st.session_state['one_query_info']['histogram_bounds']['bins'],
                        key="one_query_hist_bins",
                        on_change=lambda: update_one_query_info_bounds("histogram_bounds", "bins", st.session_state["one_query_hist_bins"]),
                    )

            if st.session_state['one_query_info']['column'] and st.session_state['one_query_info']['query_type']:
                selected_query = f"{st.session_state['one_query_info']['column']}_{st.session_state['one_query_info']['query_type'].lower()}"
                show_parameters = True
                if not is_categorical:
                    if analysis_option == "Average":
                        if (st.session_state['one_query_info']['average_bounds']['lower_bound'] is None or
                            st.session_state['one_query_info']['average_bounds']['upper_bound'] is None):
                            show_parameters = False
                    elif analysis_option == "Histogram":
                        if (st.session_state['one_query_info']['histogram_bounds']['lower_bound'] is None or
                            st.session_state['one_query_info']['histogram_bounds']['upper_bound'] is None):
                            show_parameters = False

                if show_parameters:
                    st.session_state['one_query_selected_mechanism'] = st.multiselect(
                        'Which mechanism do you want displayed',
                        ['gaussian', 'laplace'],
                        ['gaussian', 'laplace']
                    )
                    epsilon = st.slider('Privacy Parameter (\u03B5)', .01, 1.0, .25, key=f"one_query_epsilon_slider")
                    if 'gaussian' in st.session_state['one_query_selected_mechanism']:
                        delta_slider = st.slider('Global Delta (log scale)', -8, -4, -6, 1, key='one_query_delta_slider')
                        st.session_state['delta_one_query'] = 10**delta_slider
                    st.session_state['error_type'] = st.selectbox(
                        "Which error type would you like to visualize?",
                        options=['Absolute Additive Error', 'Relative Additive Error'],
                        index=1,
                        key='one_query_selected_error',
                        help=(
                            "- **Error Bounds**: Indicates predicted values' deviation from true values within 1-beta (\u03B2) confidence. \n"
                            "- **Absolute Error**: Direct difference between predicted and true values. Use for unnormalized error measurement. \n"
                            "- **Relative Error**: Absolute Error normalized by true value, useful for error comparison across scales. \n"
                            "- **Choosing Metric**: \n"
                            "   - Use **Absolute Error** for consistency in measurement units. \n"
                            "   - Choose **Relative Error** for comparative analysis across different magnitudes."
                        )
                    )
                    st.session_state['alpha'] = st.slider(
                        'beta (\u03B2) - High probability bound on accuracy',
                        0.01, .50, 0.05,
                        help='Beta represents the confidence level. A beta of 0.05 means that 95% of the hypothetical outputs will fall inside the error bars. You can see this on the chart on the left. As beta increases, more points will fall outside of the red error bounds.'
                    )
            else:
                selected_query = None

    with col2:
        st.header('Visualization')
        analysis_option = st.session_state['one_query_info']['query_type']

        # Ensure bounds are set before running visualization
        bounds_ready = False
        if analysis_option == "Average":
            bounds_ready = (
                st.session_state['one_query_info']['average_bounds']['lower_bound'] is not None
                and st.session_state['one_query_info']['average_bounds']['upper_bound'] is not None
            )
        elif analysis_option == "Histogram":
            bounds_ready = (
                st.session_state['one_query_info']['histogram_bounds']['lower_bound'] is not None
                and st.session_state['one_query_info']['histogram_bounds']['upper_bound'] is not None
            )

        if not bounds_ready and (analysis_option == "Average" or analysis_option == "Histogram"):
            st.warning("⚠️ Please enter valid lower and upper bounds before visualizing.")
        else:
            if st.session_state.one_query_selected_mechanism and st.session_state['alpha'] and st.session_state['error_type'] is not None and selected_query:
                with st.spinner('Loading Visualization...'):
                    one_query_charts = charts.one_query_privacy_accuracy_lines(
                        df,
                        selected_query,
                        st.session_state['one_query_selected_mechanism'],
                        st.session_state['alpha'],
                        st.session_state['one_query_epsilon_slider'],
                        st.session_state['error_type'],
                        st.session_state.get('delta_one_query')
                    )
                    st.plotly_chart(one_query_charts, use_container_width=True)

                with st.expander("Chart Explanations"):
                    mechanism_names = ' and '.join(st.session_state['one_query_selected_mechanism'])
                    chart_explanations = f"""
                    #### Hypothetical Outputs (Left)

                    This chart displays individual hypothetical outputs for the selected privacy-preserving mechanism(s): {mechanism_names}. 
                    The true mean is indicated by a dashed red line, serving as a benchmark to assess the accuracy of each mechanism's output. 
                    The points represent hypothetical outputs. Since differential privacy adds random noise we cannot say for certain which value will be released. 
                    The distribution of points gives a sense of where the noisy value is likely to fall.
                    Use this chart to better understand how the implementation choices will influence your noisy output.
                    This chart also gives an indication of which mechanism may be better suited for your needs. The closer the distribution to the true mean, the better the accuracy.

                    ####  Accuracy vs. Privacy Parameter (ε) (Right)

                    This chart illustrates the { st.session_state['error_type']} of the selected mechanism(s): {mechanism_names}, as a function of the privacy parameter ε.
                    As ε increases, the error bound  decreases, signifying that less stringent privacy (higher ε) correlates with higher accuracy.
                    The lines represent the error bound as specified by beta (\u03B2). The X axis can be interpreted as the true value of the query. 
                    The noisy values will fall within within the error bounds with 1-\u03B2 confidence.
                    The red dots represent the privacy/accuracy point visualized in the left chart. 
                    This chart can help decide the correct level of privacy for your data release. 
                    The error often increases exponetially as we decrease the privacy parameter. If possible, increase the epsilon beyond the 'elbow' of the curve to maximize accuracy while maintaining privacy.
                    """
                    st.markdown(chart_explanations)

def render_multiple_queries(df):
    # Corresponds to the original tab3
    st.header("Multiple Queries")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Parameters")
        st.write("Adjust the parameters to see how they affect the privacy budget.")
        k = st.slider('Number of queries', 1, 100)
        del_g = st.slider('Global Delta (log scale)', -8, -2)
        epsilon_g = st.slider('Global Epsilon', 0.01, 1.0)

    with col2:
        st.header("Visualization")
        compositors = compare(k, pow(10,del_g), .5, epsilon_g)
        dataframe = pd.DataFrame.from_dict(compositors, orient='index', columns=['Epsilon_0', 'Delta_0'])
        dataframe['Compositor'] = dataframe.index
        fig = charts.compare_compositors(dataframe)
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=0, r=0, t=30, b=0))

        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Explanation"):
            st.markdown("""
            This chart shows how different compositors affect the privacy budget.
            The x-axis represents the type of compositor used.
            The y-axis on the left represents the per query epsilon, and the y-axis on the right represents the per query delta.
            You can adjust the parameters on the left to see how they affect the privacy budget.
            """)


def render_scatterplots(df):
    # UI prototype for DP scatterplots with controls
    st.header("Scatterplots")
    col_control, col_display = st.columns([1, 3], gap="large")

    # ===== Left: Control Panel =====
    with col_control:
        st.subheader("Control Panel")

        # 1) Mechanism
        mech_enabled = st.checkbox("Mechanism", value=True, key="sc_mech_enabled")
        st.session_state["sc_mechanism"] = st.selectbox(
            "Mechanism",
            ["AHP", "DAWA", "Laplace"],
            index=1,
            key="sc_mechanism_select",
        )

        # 2) Privacy ε
        eps_enabled = st.checkbox("Privacy ε", value=True, key="sc_eps_enabled")
        st.session_state["sc_eps"] = st.slider(
            "ε",
            0.01, 1.00, 0.10, 0.01,
            key="sc_eps_slider",
            help="Placeholder ε slider; the grid on the right still shows placeholder images.",
        )

        # 3) Shift δ (log scale)
        delta_enabled = st.checkbox("Shift δ", value=False, key="sc_delta_enabled")
        power = st.slider("log10 δ", -8, -2, -6, 1, key="sc_delta_log_power")
        st.session_state["sc_delta"] = 10 ** power
        
        # 4) Bin size
        bins_enabled = st.checkbox("Bin size", value=False, key="sc_bins_enabled")
        st.session_state["sc_bins"] = st.slider(
            "Bins", 4, 128, 32, 1, key="sc_bins_slider",
            help="Placeholder bin size parameter.",
        )

        # 5) Color γ
        gamma_enabled = st.checkbox("Color γ", value=False, key="sc_gamma_enabled")
        st.session_state["sc_gamma"] = st.slider(
            "γ", 0.1, 3.0, 1.0, 0.1, key="sc_gamma_slider",
            help="Placeholder color/contrast γ parameter.",
        )

        # original image
        st.image(
            "image.png",
            caption="Original Graph",
            use_container_width=False,
        )

    # ===== Right: Visualization Grid =====
    with col_display:
        epsilons = ["ε=0.5", "ε=0.1", "ε=0.05", "ε=0.01"]
        mechanisms = ["AHP", "DAWA", "Laplace"]

        # Header (epsilon columns)
        header_cols = st.columns(len(epsilons) + 1)
        header_cols[0].markdown("### ")
        for i, eps in enumerate(epsilons):
            header_cols[i + 1].markdown(f"### {eps}")

        # Rows: mechanisms
        for mech in mechanisms:
            row_cols = st.columns(len(epsilons) + 1)
            row_cols[0].markdown(f"### {mech}")
            for i, _ in enumerate(epsilons):
                with row_cols[i + 1]:
                    # Placeholder image; replace with real plots later
                    st.image(
                        "image.png",
                        use_container_width=True,
                    )


section = st.radio(
    "Select Section",
    ["Simulations", "One Query", "Multiple Queries", "Scatterplots"],
    horizontal=True,
    key="section_selector",
)

if section == "Simulations":
    render_simulations(st.session_state['selected_dataset'])
elif section == "One Query":
    render_one_query(st.session_state['selected_dataset'])
elif section == "Multiple Queries":
    render_multiple_queries(st.session_state['selected_dataset'])
elif section == "Scatterplots":
    render_scatterplots(st.session_state['selected_dataset'])
