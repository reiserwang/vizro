#!/usr/bin/env python3
"""
Causal Intervention Analysis Module
Handles 'what-if' scenario analysis using do-calculus.
"""

import pandas as pd
import numpy as np
try:
    import gradio as gr
    _progress = gr.Progress()
except ImportError:
    gr = None
    _progress = None
import html

from core import dashboard_config
from .causal_network_utils import has_cycles, resolve_cycles
from .causal_discretization import create_ultra_robust_split_points

def perform_causal_intervention_analysis(target_var, intervention_var, intervention_value, progress=_progress):
    """Perform causal intervention analysis (do-calculus)"""

    # Lazy-import causalnex: its pytorch subpackage imports torch unconditionally
    # which would crash the frozen native app (torch not bundled). Import here
    # so PyInstaller's static analyser never sees the torch dependency chain.
    try:
        from causalnex.structure.notears import from_pandas
        from causalnex.network import BayesianNetwork
        from causalnex.inference import InferenceEngine
        from causalnex.discretiser import Discretiser
    except ImportError as _e:
        def _unavailable(*a, **kw):
            return (
                "❌ Causal intervention requires additional libraries not included in the native app."
                " Please run in web mode: python main.py --mode ui",
                f"Missing dependency: {_e}"
            )
        return _unavailable()

    try:
        progress(0.1, desc="🔬 Preparing intervention analysis...")

        if dashboard_config.current_data is None:
            return "❌ No data loaded", "Please upload data first"

        # Get numeric data
        df_numeric = dashboard_config.current_data.select_dtypes(include=[np.number])
        if df_numeric.empty:
            return "❌ No numeric data found", "Please ensure your data contains numeric variables"

        if target_var not in df_numeric.columns or intervention_var not in df_numeric.columns:
            return "❌ Variables not found", "Selected variables not found in data"

        progress(0.4, desc="🔍 Validating data quality...")

        # Check for sufficient variation in key variables
        target_variation = df_numeric[target_var].std()
        intervention_variation = df_numeric[intervention_var].std()

        if target_variation < 1e-10:
            return f"""
            ❌ Intervention analysis failed: Insufficient variation in target variable

            **Problem:** The target variable '{html.escape(str(target_var))}' has no variation (std = {target_variation:.2e})

            **Solutions:**
            • Choose a different target variable with more diverse values
            • Check if the data contains only constant values
            • Ensure the variable represents a meaningful outcome
            """, "Target variable has insufficient variation"

        if intervention_variation < 1e-10:
            return f"""
            ❌ Intervention analysis failed: Insufficient variation in intervention variable

            **Problem:** The intervention variable '{html.escape(str(intervention_var))}' has no variation (std = {intervention_variation:.2e})

            **Solutions:**
            • Choose a different intervention variable with more diverse values
            • Check if the data contains only constant values
            • Ensure the variable can be meaningfully changed
            """, "Intervention variable has insufficient variation"

        progress(0.3, desc="🏗️ Building causal structure...")

        # Build causal structure using NOTEARS with cycle detection
        try:
            sm = from_pandas(df_numeric, max_iter=100, h_tol=1e-8, w_threshold=0.3)

            # Check for cycles in the structure
            if has_cycles(sm):
                print("⚠️ Detected cycles in causal structure, applying cycle resolution...")
                sm = resolve_cycles(sm, df_numeric)

        except Exception as e:
            if "not acyclic" in str(e) or "cycle" in str(e).lower():
                return f"""
                ❌ Intervention analysis failed: Cyclic causal structure detected

                **Problem:** The causal discovery algorithm found bidirectional relationships that create cycles.

                **Detected Issue:** {html.escape(str(e))}

                **Solutions:**
                • Try with fewer variables (select 5-10 most important ones)
                • Increase the w_threshold parameter to filter weak relationships
                • Use domain knowledge to remove variables that shouldn't be causally related
                • Consider that some relationships might be correlational rather than causal

                **Technical Note:** Bayesian Networks require acyclic structures (DAGs).
                Cycles often indicate confounding variables or bidirectional relationships
                that need to be resolved through domain expertise.
                """, f"Cyclic structure error: {str(e)}"
            else:
                raise e

        progress(0.5, desc="🧠 Creating Bayesian Network...")

        # Create split points for each column with extensive validation
        split_points = {}
        failed_columns = []

        for col in df_numeric.columns:
            try:
                # Get column data and check quality
                col_data = df_numeric[col]
                unique_count = col_data.nunique()

                print(f"📊 Processing {col}: {len(col_data)} values, {unique_count} unique")

                if unique_count < 2:
                    print(f"⚠️ Column {col} has insufficient variation ({unique_count} unique values)")
                    failed_columns.append(col)
                    continue

                splits = create_ultra_robust_split_points(col_data)

                # Validate splits
                if len(splits) != 2:
                    raise ValueError(f"Expected 2 split points, got {len(splits)}")

                if splits[1] <= splits[0]:
                    raise ValueError(f"Split points not monotonic: {splits}")

                if not all(np.isfinite(splits)):
                    raise ValueError(f"Split points contain invalid values: {splits}")

                split_points[col] = splits
                print(f"✅ Created valid split points for {col}: {splits}")

            except Exception as e:
                print(f"❌ Failed to create split points for {col}: {e}")
                failed_columns.append(col)

        # Remove failed columns from analysis
        if failed_columns:
            print(f"⚠️ Removing {len(failed_columns)} columns with discretization issues: {failed_columns}")
            df_numeric = df_numeric.drop(columns=failed_columns)

            # Check if we still have enough variables
            if len(df_numeric.columns) < 2:
                return f"""
                ❌ Intervention analysis failed: Insufficient valid variables

                **Problem:** After removing columns with discretization issues, only {len(df_numeric.columns)} variables remain.

                **Failed columns:** {', '.join([html.escape(str(c)) for c in failed_columns])}

                **Solutions:**
                • Use a dataset with more diverse numeric variables
                • Check data quality (remove constant or near-constant columns)
                • Ensure variables have at least 10+ unique values
                • Try with different variable combinations
                """, "Insufficient variables for analysis"

            # Check if target/intervention variables are still available
            if target_var not in df_numeric.columns:
                return f"""
                ❌ Intervention analysis failed: Target variable removed

                **Problem:** Target variable '{html.escape(str(target_var))}' was removed due to discretization issues.

                **Solutions:**
                • Choose a different target variable with more variation
                • Check if '{target_var}' has sufficient unique values
                • Ensure the variable represents a meaningful outcome
                """, "Target variable unavailable"

            if intervention_var not in df_numeric.columns:
                return f"""
                ❌ Intervention analysis failed: Intervention variable removed

                **Problem:** Intervention variable '{html.escape(str(intervention_var))}' was removed due to discretization issues.

                **Solutions:**
                • Choose a different intervention variable with more variation
                • Check if '{intervention_var}' has sufficient unique values
                • Ensure the variable can be meaningfully changed
                """, "Intervention variable unavailable"

        # Final validation and cleanup of all split points before creating discretizer
        print(f"🔍 Final validation of split points for {len(split_points)} variables...")
        cleaned_split_points = {}

        for col, splits in split_points.items():
            # Round to avoid floating point precision issues
            rounded_splits = [round(float(s), 10) for s in splits]

            # Ensure strict monotonic increasing with minimum separation
            if len(rounded_splits) != 2:
                print(f"❌ Invalid number of splits for {col}: {len(rounded_splits)}")
                continue

            if not all(np.isfinite(rounded_splits)):
                print(f"❌ Non-finite splits for {col}: {rounded_splits}")
                continue

            # Ensure monotonic with minimum separation
            min_separation = 1e-8
            if rounded_splits[1] <= rounded_splits[0]:
                print(f"⚠️ Non-monotonic splits for {col}: {rounded_splits}, fixing...")
                rounded_splits[1] = rounded_splits[0] + min_separation
            elif rounded_splits[1] - rounded_splits[0] < min_separation:
                print(f"⚠️ Insufficient separation for {col}: {rounded_splits}, fixing...")
                rounded_splits[1] = rounded_splits[0] + min_separation

            cleaned_split_points[col] = rounded_splits
            print(f"✅ Validated splits for {col}: {rounded_splits} (diff: {rounded_splits[1] - rounded_splits[0]:.2e})")

        # Update split_points with cleaned version
        split_points = cleaned_split_points

        # Discretize data
        print(f"🏗️ Discretizing data...")
        df_discretised = df_numeric.copy()

        try:
            for col in df_numeric.columns:
                # Simple quantile-based discretization
                q33 = df_numeric[col].quantile(0.33)
                q67 = df_numeric[col].quantile(0.67)

                df_discretised[col] = pd.cut(
                    df_numeric[col],
                    bins=[-np.inf, q33, q67, np.inf],
                    labels=['low', 'medium', 'high']
                )
                print(f"✅ Discretized {col}: low ≤ {q33:.3f}, medium ≤ {q67:.3f}, high > {q67:.3f}")

        except Exception as e:
            return f"""
            ❌ Intervention analysis failed: Data transformation error

            **Problem:** Manual discretization failed

            **Error:** {html.escape(str(e))}

            **Solutions:**
            • Check that your data contains valid numeric values
            • Remove any infinite or extremely large values
            • Ensure variables have reasonable ranges
            • Try with a smaller subset of variables
            • Consider using different variable combinations

            **Data info:**
            • Target variable range: {df_numeric[target_var].min():.3f} to {df_numeric[target_var].max():.3f}
            • Intervention variable range: {df_numeric[intervention_var].min():.3f} to {df_numeric[intervention_var].max():.3f}
            """, f"Data transformation error: {str(e)}"

        # Create Bayesian Network
        bn = BayesianNetwork(sm)
        bn = bn.fit_node_states(df_discretised)
        bn = bn.fit_cpds(df_discretised, method="BayesianEstimator", bayes_prior="K2")

        progress(0.7, desc="🎯 Performing intervention...")

        # Create inference engine
        ie = InferenceEngine(bn)

        # Validate and discretize intervention value
        intervention_min = df_numeric[intervention_var].min()
        intervention_max = df_numeric[intervention_var].max()

        # Check if intervention value is within reasonable range
        if intervention_value < intervention_min * 0.5 or intervention_value > intervention_max * 2.0:
            return f"""
            ❌ Intervention analysis failed: Intervention value out of range

            **Problem:** Intervention value {html.escape(str(intervention_value))} is outside reasonable range

            **Data range for {html.escape(str(intervention_var))}:**
            • Minimum: {intervention_min:.3f}
            • Maximum: {intervention_max:.3f}
            • Suggested range: {intervention_min:.3f} to {intervention_max:.3f}

            **Solutions:**
            • Use an intervention value within the data range
            • Try values between {intervention_min:.1f} and {intervention_max:.1f}
            • Consider the practical meaning of your intervention
            """, "Intervention value out of range"

        try:
            # Use the same manual discretization approach for intervention value
            q33 = df_numeric[intervention_var].quantile(0.33)
            q67 = df_numeric[intervention_var].quantile(0.67)

            if intervention_value <= q33:
                intervention_state = 'low'
            elif intervention_value <= q67:
                intervention_state = 'medium'
            else:
                intervention_state = 'high'

            print(f"✅ Intervention value {intervention_value} discretized to state: {intervention_state}")
            print(f"📊 Discretization thresholds: low ≤ {q33:.3f}, medium ≤ {q67:.3f}, high > {q67:.3f}")

        except Exception as e:
            return f"""
            ❌ Intervention analysis failed: Could not discretize intervention value

            **Problem:** Error discretizing intervention value {html.escape(str(intervention_value))}: {html.escape(str(e))}

            **Solutions:**
            • Try an intervention value closer to the data mean: {df_numeric[intervention_var].mean():.3f}
            • Ensure the value is a valid number
            • Check that the value makes sense for your variable
            """, f"Intervention discretization error: {str(e)}"

        # Perform intervention (do-calculus)
        intervention_query = ie.do_intervention(
            {intervention_var: intervention_state},
            {target_var: list(df_discretised[target_var].unique())}
        )

        progress(0.9, desc="📊 Generating results...")

        # Calculate baseline probabilities (without intervention)
        baseline_query = ie.query({target_var: list(df_discretised[target_var].unique())})

        # Create results summary
        safe_target_var = html.escape(str(target_var))
        safe_intervention_var = html.escape(str(intervention_var))
        safe_intervention_value = html.escape(str(intervention_value))
        safe_intervention_state = html.escape(str(intervention_state))

        results_html = f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🎯 Causal Intervention Analysis</h3>

            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📋 Analysis Setup</h4>
                <p><strong>Target Variable:</strong> {safe_target_var}</p>
                <p><strong>Intervention Variable:</strong> {safe_intervention_var}</p>
                <p><strong>Intervention Value:</strong> {safe_intervention_value} → {safe_intervention_state}</p>
            </div>

            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📊 Probability Distributions</h4>

                <h5>🔵 Baseline (Observational)</h5>
                <div style="font-family: monospace; background: white; padding: 10px; border-radius: 4px;">
        """

        for state, prob in baseline_query.items():
            safe_state = html.escape(str(state))
            results_html += f"P({safe_target_var} = {safe_state}) = {prob:.4f}<br>"

        results_html += """
                </div>

                <h5>🔴 After Intervention</h5>
                <div style="font-family: monospace; background: white; padding: 10px; border-radius: 4px;">
        """

        for state, prob in intervention_query.items():
            safe_state = html.escape(str(state))
            results_html += f"P({safe_target_var} = {safe_state} | do({safe_intervention_var} = {safe_intervention_state})) = {prob:.4f}<br>"

        results_html += """
                </div>
            </div>

            <div style="background: #f3e5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>📈 Causal Effect Analysis</h4>
        """

        # Calculate causal effects
        for state in baseline_query.keys():
            safe_state = html.escape(str(state))
            baseline_prob = baseline_query[state]
            intervention_prob = intervention_query[state]
            effect = intervention_prob - baseline_prob
            effect_pct = (effect / baseline_prob * 100) if baseline_prob > 0 else 0

            effect_color = "#4caf50" if effect > 0 else "#f44336" if effect < 0 else "#757575"
            results_html += f"""
                <p style="color: {effect_color};">
                    <strong>{safe_target_var} = {safe_state}:</strong>
                    {effect:+.4f} ({effect_pct:+.1f}% change)
                </p>
            """

        results_html += f"""
            </div>

            <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>💡 Interpretation</h4>
                <p>This analysis shows the <strong>causal effect</strong> of intervening on <strong>{safe_intervention_var}</strong>
                and setting it to <strong>{safe_intervention_value}</strong> on the probability distribution of <strong>{safe_target_var}</strong>.</p>
                <p>The differences between baseline and intervention probabilities represent the <strong>true causal impact</strong>,
                not just correlation.</p>
            </div>
        </div>
        """

        progress(1.0, desc="✅ Intervention analysis complete!")

        return results_html, "✅ Causal intervention analysis completed successfully!"

    except Exception as e:
        progress(1.0, desc="❌ Analysis failed")

        # Provide specific error guidance
        error_details = html.escape(str(e))
        if "monotonically increasing" in str(e):
            error_msg = """
            ❌ Intervention analysis failed: Data discretization issue

            **Problem:** The selected variables have insufficient variation for Bayesian Network analysis.

            **Solutions:**
            • Try different variables with more variation
            • Ensure your data has diverse values (not mostly the same)
            • Use variables with continuous distributions
            • Check that your intervention value is within the data range

            **Technical details:** """ + error_details
        elif "intervention" in str(e).lower():
            error_msg = f"""
            ❌ Intervention analysis failed: {error_details}

            **Suggestions:**
            • Check that the intervention value is reasonable for your data
            • Ensure both variables have sufficient data points
            • Try with different variable combinations
            """
        else:
            error_msg = f"""
            ❌ Intervention analysis failed: {error_details}

            **Common causes:**
            • Insufficient data (need at least 50+ rows)
            • Variables with little variation
            • Missing or invalid values
            • Causal structure too complex for the data size
            """

        return error_msg, error_msg
