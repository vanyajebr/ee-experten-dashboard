import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="EE-Experten Dashboard", layout="wide")

agent_list = [
    "All Agents",
    "Oleg Akram", "Sylvia Marquart", "Martina Schneble", "Anastasiia Mihovych", "Stefan Renz",
    "Ute Schorpp", "Roland M. Schwarz", "Sinan Turgut", "Bettina Czech", "Christa Hanke",
    "Annette Muller", "Nathalie Munz-Vecchio", "Cindy Cremer"
]
agent_list_no_all = agent_list[1:]
product_cols = ['ea', 'em', 'isfp', 'hlb']

page = st.sidebar.selectbox("Select Page", ["Dashboard", "Agent Comparison"])
if page == "Dashboard":
    selected_agent = st.sidebar.selectbox("Select Agent", agent_list)
else:
    selected_agent = None

st.sidebar.header("📁 Upload Data")
sales_file = st.sidebar.file_uploader("Upload Sales Data CSV", type="csv")
calls_file = st.sidebar.file_uploader("Upload Calls Data CSV", type="csv")

def filter_period(df, datecol, selected_date):
    if selected_date == "All Time":
        return df.copy()
    else:
        start = pd.Timestamp(f"{selected_date.split()[1]}-{pd.Timestamp(selected_date.split()[0] + ' 1').month:02d}-01 00:00:00", tz="Europe/Berlin")
        end = (start + pd.offsets.MonthEnd()).replace(hour=23, minute=59, second=59)
        return df[(df[datecol] >= start) & (df[datecol] <= end)]

def exclude_weekends(df, dt_col):
    df['weekday'] = pd.to_datetime(df[dt_col], errors='coerce').dt.weekday
    return df[~df['weekday'].isin([5, 6])].copy()

def make_product_int_cols(df):
    for col in product_cols:
        if col in df.columns:
            df[col + '_int_only'] = df[col].apply(
                lambda x: int(x) if pd.notnull(x) and float(x).is_integer() else 0
            )
    return df

if sales_file and calls_file:
    df_sales = pd.read_csv(io.StringIO(sales_file.getvalue().decode("utf-8")), low_memory=False)
    df_calls = pd.read_csv(io.StringIO(calls_file.getvalue().decode("utf-8")), low_memory=False)

    # --- Robust Date Parsing with utc=True ---
    df_sales['createdate'] = pd.to_datetime(df_sales['createdate'], dayfirst=True, errors='coerce')
    df_sales['createdate_berlin'] = pd.to_datetime(df_sales['createdate'], utc=True).dt.tz_convert('Europe/Berlin')
    df_calls['Start time'] = pd.to_datetime(df_calls['Start time'], errors='coerce')
    df_calls['Start time'] = pd.to_datetime(df_calls['Start time'], utc=True).dt.tz_convert('Europe/Berlin')

    df_sales = make_product_int_cols(df_sales)

    date_options = ["All Time", "December 2024", "January 2025", "February 2025", "March 2025", "April 2025"]
    selected_date = st.selectbox("Select Date Range:", date_options)

    if page == "Dashboard":
        st.title("📊 EE-Experten Sales & Calls Dashboard")

        df_team_period = filter_period(df_sales, 'createdate_berlin', selected_date)
        df_team_calls_period = filter_period(df_calls, 'Start time', selected_date)

        if selected_agent == "All Agents":
            df_filtered = df_team_period.copy()
            df_calls_filtered = df_team_calls_period.copy()
        else:
            df_filtered = df_team_period[df_team_period['createuser'] == selected_agent].copy()
            df_calls_filtered = df_team_calls_period[df_team_calls_period['User'] == selected_agent].copy()

        df_filtered['createdate_berlin'] = pd.to_datetime(df_filtered['createdate_berlin'], errors='coerce')
        df_calls_filtered['Start time'] = pd.to_datetime(df_calls_filtered['Start time'], errors='coerce')
        df_filtered = make_product_int_cols(df_filtered)

        df_filtered['full_name'] = (
            df_filtered['kunde_vorname'].astype(str).str.strip() + " " +
            df_filtered['kunde_nachname'].astype(str).str.strip()
        )

        product_cols_int = [col + '_int_only' for col in product_cols if col + '_int_only' in df_filtered.columns]
        total_products_sold = df_filtered[product_cols_int].sum().sum()
        total_stornos = df_filtered[df_filtered['storno'] == 1].shape[0]
        # Consistent logic for all stats/graphs
        total_products_sold_no_storno = total_products_sold - total_stornos
        total_deals = df_filtered[['full_name']].drop_duplicates().shape[0]
        storno_rate = (total_stornos / total_products_sold * 100) if total_products_sold else 0

        total_products_no_storno_team_period = (
            filter_period(df_sales, 'createdate_berlin', selected_date)
            [product_cols_int].sum().sum()
            - filter_period(df_sales, 'createdate_berlin', selected_date)[filter_period(df_sales, 'createdate_berlin', selected_date)['storno'] == 1].shape[0]
        )
        percent_of_team_sales = (
            "100.0%" if selected_agent == "All Agents" else
            f"{(total_products_sold_no_storno / total_products_no_storno_team_period * 100):.1f}%" if total_products_no_storno_team_period else "—"
        )

        call_df = df_calls_filtered.copy()
        call_df['date'] = call_df['Start time'].dt.date
        call_df = exclude_weekends(call_df, 'Start time')
        calls_per_day = call_df.groupby('date').size()
        avg_calls_per_day = f"{calls_per_day.mean():.2f}" if not calls_per_day.empty else "—"

        call_df['over3min'] = call_df['Duration'] > 180
        calls3min_per_day = call_df[call_df['over3min']].groupby('date').size()
        avg_calls3min_per_day = f"{calls3min_per_day.mean():.2f}" if not calls3min_per_day.empty else "—"

        df_filtered['day'] = df_filtered['createdate_berlin'].dt.date
        df_filtered = exclude_weekends(df_filtered, 'createdate_berlin')
        sales_per_day = df_filtered.groupby('day')[product_cols_int].sum()
        stornos_per_day = df_filtered.groupby('day').apply(lambda x: (x['storno'] == 1).sum())
        sales_no_storno_per_day = sales_per_day.sum(axis=1) - stornos_per_day
        avg_product_sold_per_day = f"{sales_no_storno_per_day.mean():.2f}" if hasattr(sales_no_storno_per_day, 'mean') and not sales_no_storno_per_day.empty else "—"

        sale_times = pd.to_datetime(df_filtered['createdate_berlin'], errors='coerce').dt.time.dropna()
        if not sale_times.empty:
            mins_since_midnight = [t.hour * 60 + t.minute for t in sale_times]
            avg_min = int(np.mean(mins_since_midnight))
            avg_hour = avg_min // 60
            avg_minute = avg_min % 60
            avg_sale_time_of_day = f"{avg_hour:02d}:{avg_minute:02d}"
        else:
            avg_sale_time_of_day = "—"

        if selected_agent == "All Agents":
            avg_break_between_calls = "—"
        else:
            if len(call_df) > 1:
                call_df = call_df.sort_values(by="Start time")
                mean_gaps = []
                for date, group in call_df.groupby('date'):
                    call_times = pd.to_datetime(group['Start time'], errors='coerce').sort_values()
                    if len(call_times) < 2:
                        continue
                    time_diffs = call_times.diff().dropna().dt.total_seconds() / 60
                    mean_gaps.append(time_diffs.mean())
                if mean_gaps:
                    avg_break_between_calls = f"{float(np.mean(mean_gaps)):.1f} min"
                else:
                    avg_break_between_calls = "—"
            else:
                avg_break_between_calls = "—"

        st.markdown("### 🔢 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Deals (Clients)", total_deals)
        col2.metric("Total Products Sold", total_products_sold)
        col3.metric("Total Stornos", total_stornos)
        col4.metric("Total Products Sold (No Storno)", total_products_sold_no_storno)
        col5.metric("Storno Rate", f"{storno_rate:.2f}%")

        unique_called_numbers = df_calls_filtered['Called number'].dropna().nunique()
        calls_above_3min = df_calls_filtered[df_calls_filtered['Duration'] > 180]['Called number'].dropna().nunique()
        calls_3min = df_calls_filtered[df_calls_filtered['Duration'] > 180]
        avg_call_duration = calls_3min['Duration'].mean() / 60 if not calls_3min['Duration'].empty else 0
        sales_rate = (total_deals / unique_called_numbers) * 100 if unique_called_numbers else 0
        col6, col7, col8, col9, col10 = st.columns(5)
        col6.metric("Unique Numbers Called", unique_called_numbers)
        col7.metric("Calls > 3 Min", calls_above_3min)
        col8.metric("Avg Call Duration >3min (min)", f"{avg_call_duration:.2f}")
        col9.metric("% of Team Sales", percent_of_team_sales)
        col10.metric("Sales Rate", f"{sales_rate:.2f}%")

        col11, col12, col13, col14, col15 = st.columns(5)
        col11.metric("Avg Product Sold/Day (No Storno)", avg_product_sold_per_day)
        col12.metric("Avg Calls/Day", avg_calls_per_day)
        col13.metric("Avg Calls >3min/Day", avg_calls3min_per_day)
        col14.metric("Avg Sale Time (HH:MM)", avg_sale_time_of_day)
        col15.metric("Avg Break Between Calls", avg_break_between_calls)

        # --- Prepare line chart data for All Time or monthly ---
        manual_order = ["2024-12", "2025-01", "2025-02", "2025-03", "2025-04"]
        if selected_date == "All Time":
            df_filtered['month_period'] = pd.to_datetime(df_filtered['createdate_berlin'], errors='coerce').dt.to_period('M')
            df_calls_filtered['month_period'] = pd.to_datetime(df_calls_filtered['Start time'], errors='coerce').dt.to_period('M')

            # Consistent: Products Sold (No Storno) per month = sum(products) - sum(stornos)
            monthly_products = df_filtered.groupby('month_period')[product_cols_int].sum().sum(axis=1)
            monthly_stornos = df_filtered.groupby('month_period').apply(lambda x: (x['storno'] == 1).sum())
            monthly_products_sold_no_storno = monthly_products - monthly_stornos

            monthly_products_sold_no_storno = monthly_products_sold_no_storno.reindex(manual_order, fill_value=0)
            monthly_products_sold_no_storno.index = pd.CategoricalIndex(monthly_products_sold_no_storno.index, categories=manual_order, ordered=True)
            st.subheader("Products Sold (No Storno) Over Time (Monthly)")
            st.line_chart(monthly_products_sold_no_storno)

            # Sales Rate graph (monthly)
            deals_per_period = (
                df_filtered
                .groupby('month_period')[['full_name']]
                .nunique()['full_name']
            )
            unique_numbers_per_period = df_calls_filtered.groupby('month_period')['Called number'].nunique()
            sales_rate_per_period = (deals_per_period / unique_numbers_per_period * 100).fillna(0)
            sales_rate_per_period = sales_rate_per_period.reindex(manual_order, fill_value=0)
            sales_rate_per_period.index = pd.CategoricalIndex(sales_rate_per_period.index, categories=manual_order, ordered=True)
            st.subheader("Sales Rate Over Time (Monthly)")
            st.line_chart(sales_rate_per_period)

            # Storno Rate Over Time
            total_products_per_period = df_filtered.groupby('month_period')[product_cols_int].sum().sum(axis=1)
            storno_per_period = df_filtered[df_filtered['storno'] == 1].groupby('month_period').size()
            storno_rate_per_period = (storno_per_period / total_products_per_period * 100).fillna(0)
            storno_rate_per_period = storno_rate_per_period.reindex(manual_order, fill_value=0)
            storno_rate_per_period.index = pd.CategoricalIndex(storno_rate_per_period.index, categories=manual_order, ordered=True)
            if selected_agent == "All Agents":
                st.subheader("Storno Rate Over Time (Monthly)")
                st.line_chart(storno_rate_per_period)

            # Unique Numbers Called
            unique_numbers_per_period = df_calls_filtered.groupby('month_period')['Called number'].nunique()
            unique_numbers_per_period = unique_numbers_per_period.reindex(manual_order, fill_value=0)
            unique_numbers_per_period.index = pd.CategoricalIndex(unique_numbers_per_period.index, categories=manual_order, ordered=True)
            st.subheader("Unique Numbers Called Over Time (Monthly)")
            st.line_chart(unique_numbers_per_period)
        else:
            # Daily, weekends excluded
            df_filtered = exclude_weekends(df_filtered, 'createdate_berlin')
            df_calls_filtered = exclude_weekends(df_calls_filtered, 'Start time')
            df_filtered['day'] = pd.to_datetime(df_filtered['createdate_berlin'], errors='coerce').dt.day
            df_calls_filtered['day'] = pd.to_datetime(df_calls_filtered['Start time'], errors='coerce').dt.day

            # Consistent: Products Sold (No Storno) per day = sum(products) - sum(stornos)
            daily_products = df_filtered.groupby('day')[product_cols_int].sum().sum(axis=1)
            daily_stornos = df_filtered.groupby('day').apply(lambda x: (x['storno'] == 1).sum())
            daily_products_sold_no_storno = daily_products - daily_stornos

            st.subheader(f"Products Sold (No Storno) Over Time (Daily, excl. weekends) - {selected_date}")
            st.line_chart(daily_products_sold_no_storno)

            # Only show sales rate graph for all agents (monthly); daily not shown for individuals
            unique_numbers_per_period = df_calls_filtered.groupby('day')['Called number'].nunique()
            st.subheader(f"Unique Numbers Called Over Time (Daily, excl. weekends) - {selected_date}")
            st.line_chart(unique_numbers_per_period)

        # --- Product Distribution Pie Chart (use same filter as metrics) ---
        pie_units = df_filtered[product_cols_int].sum().sum() if len(df_filtered) > 0 else 0
        pie_stornos = df_filtered[df_filtered['storno'] == 1].shape[0]
        pie_no_storno = pie_units - pie_stornos
        product_sums = df_filtered[product_cols_int].sum()
        if product_sums.sum() > 0:
            pie_values = (product_sums / product_sums.sum() * pie_no_storno).round(0).astype(int)
        else:
            pie_values = [0] * len(product_cols_int)
        product_labels = [col.replace('_int_only', '') for col in product_cols_int]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(pie_values, labels=product_labels, autopct=lambda p: f'{p:.1f}% ({int(p * sum(pie_values) / 100)})')
        ax.set_title("Product Distribution")
        st.pyplot(fig)

    # --- AGENT COMPARISON ---
    elif page == "Agent Comparison":
        st.title("👥 Agent Comparison")
        df_sales['createdate_berlin'] = pd.to_datetime(df_sales['createdate_berlin'], errors='coerce')
        df_calls['Start time'] = pd.to_datetime(df_calls['Start time'], errors='coerce')
        df_sales = make_product_int_cols(df_sales)
        df_period = filter_period(df_sales, 'createdate_berlin', selected_date)
        df_calls_period = filter_period(df_calls, 'Start time', selected_date)
        df_period = make_product_int_cols(df_period)

        product_cols_int = [col + '_int_only' for col in product_cols if col + '_int_only' in df_period.columns]
        team_products = df_period[product_cols_int].sum().sum()
        team_stornos = df_period[df_period['storno'] == 1].shape[0]
        total_team_products_no_storno = team_products - team_stornos

        comparison_data = []
        for agent in agent_list_no_all:
            sales_agent = df_period[df_period['createuser'] == agent].copy()
            calls_agent = df_calls_period[df_calls_period['User'] == agent].copy()
            sales_agent = make_product_int_cols(sales_agent)
            products = sales_agent[product_cols_int].sum().sum()
            stornos = sales_agent[sales_agent['storno'] == 1].shape[0]
            products_no_storno = products - stornos
            percent_of_team_sales = (products_no_storno / total_team_products_no_storno * 100) if total_team_products_no_storno else 0

            sales_agent['full_name'] = (
                sales_agent['kunde_vorname'].astype(str).str.strip() + " " +
                sales_agent['kunde_nachname'].astype(str).str.strip()
            )
            total_deals = sales_agent[['full_name']].drop_duplicates().shape[0]
            unique_called_numbers = calls_agent['Called number'].dropna().nunique()
            calls_above_3min = calls_agent[calls_agent['Duration'] > 180]['Called number'].dropna().nunique()
            calls_3min = calls_agent[calls_agent['Duration'] > 180]
            avg_call_duration = calls_3min['Duration'].mean() / 60 if not calls_3min['Duration'].empty else 0
            sales_rate = (total_deals / unique_called_numbers) * 100 if unique_called_numbers else 0

            calls_agent['date'] = pd.to_datetime(calls_agent['Start time'], errors='coerce').dt.date
            calls_agent = exclude_weekends(calls_agent, 'Start time')
            calls_per_day = calls_agent.groupby('date').size()
            avg_calls_per_day = float(calls_per_day.mean()) if not calls_per_day.empty else 0
            calls_agent['over3min'] = calls_agent['Duration'] > 180
            calls3min_per_day = calls_agent[calls_agent['over3min']].groupby('date').size()
            avg_calls3min_per_day = float(calls3min_per_day.mean()) if not calls3min_per_day.empty else 0

            sales_agent['day'] = pd.to_datetime(sales_agent['createdate_berlin'], errors='coerce').dt.date
            sales_agent = exclude_weekends(sales_agent, 'createdate_berlin')
            sales_per_day = sales_agent.groupby('day')[product_cols_int].sum()
            stornos = sales_agent[sales_agent['storno'] == 1].shape[0]
            stornos_per_day = sales_agent.groupby('day').apply(lambda x: (x['storno'] == 1).sum())
            sales_no_storno_per_day = sales_per_day.sum(axis=1) - stornos_per_day
            avg_product_sold_per_day = float(sales_no_storno_per_day.mean()) if hasattr(sales_no_storno_per_day, 'mean') and not sales_no_storno_per_day.empty else 0

            product_sum = sales_agent[product_cols_int].sum().sum()
            total_stornos = sales_agent[sales_agent['storno'] == 1].shape[0]
            storno_rate = (total_stornos / product_sum * 100) if product_sum else 0

            comparison_data.append({
                "Agent": agent,
                "% of Team Sales": percent_of_team_sales,
                "Total Deals": total_deals,
                "Total Products Sold (No Storno)": products_no_storno,
                "Storno Rate": storno_rate,
                "Unique Numbers Called": unique_called_numbers,
                "Calls >3min": calls_above_3min,
                "Avg Calls/Day": avg_calls_per_day,
                "Avg Calls >3min/Day": avg_calls3min_per_day,
                "Avg Product Sold/Day (No Storno)": avg_product_sold_per_day,
                "Sales Rate": sales_rate,
                "Avg Call Duration >3min (min)": avg_call_duration,
            })

        df_compare = pd.DataFrame(comparison_data).set_index("Agent")
        compare_metrics = [
            "% of Team Sales", "Total Deals", "Total Products Sold (No Storno)",
            "Storno Rate", "Unique Numbers Called", "Calls >3min", "Avg Calls/Day",
            "Avg Calls >3min/Day", "Avg Product Sold/Day (No Storno)",
            "Sales Rate", "Avg Call Duration >3min (min)"
        ]
        st.write(f"### Bar Chart Comparison ({selected_date})")
        for metric in compare_metrics:
            st.subheader(metric)
            # Matplotlib sorted bar chart for always descending values
            sorted_df = df_compare[metric].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(sorted_df.index, sorted_df.values)
            ax.set_ylabel(metric)
            ax.set_xlabel("Agent")
            ax.set_xticklabels(sorted_df.index, rotation=40, ha='right')
            for i, v in enumerate(sorted_df.values):
                ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
            st.pyplot(fig)

else:
    st.warning("⚠️ Please upload both Sales and Calls CSV files to begin.")
