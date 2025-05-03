# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 23:21:57 2025

@author: Qiong Wu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 20:00:00 2025

@author: Grok
"""
# Clear all variables in the current Python environment
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Confirm variables are cleared
print("All local variables have been cleared.")

# Clear console


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


clear_console()
"-----------------------------------------------------------------------------"
"-----------------------------------------------------------------------------"
"-----------------------------------------------------------------------------"

# ABS Parameters
pool_start_balance = 100_000_000  # $100 million
num_loans = 5000
avg_loan_size = 20_000  # $20,000
loan_term_months = 60  # 5 years
loan_rate = 0.045  # 4.5%
servicing_fee = 0.005  # 0.5% annual
overcollateralization = 2_000_000  # $2 million cushion

# Tranche structure
senior_start_balance = 90_000_000  # $90 million
mezzanine_start_balance = 7_000_000  # $7 million
subordinate_start_balance = 2_000_000  # $2 million
equity_start_balance = 1_000_000  # $1 million
senior_coupon_net = 0.025  # 2.5%
mezzanine_coupon_net = 0.04  # 4.0%
subordinate_coupon_net = 0.06  # 6.0%
equity_coupon_net = 0.08  # 8.0%

# Monthly rates
loan_monthly_rate = loan_rate / 12
senior_monthly_coupon = senior_coupon_net / 12
mezzanine_monthly_coupon = mezzanine_coupon_net / 12
subordinate_monthly_coupon = subordinate_coupon_net / 12
equity_monthly_coupon = equity_coupon_net / 12

# Assumptions for prepayment and default
cpr_annual = 0.10  # 10% constant prepayment rate
cdr_annual = 0.02  # 2% constant default rate
loss_severity = 0.30  # 30% loss severity
monthly_cpr = 1 - (1 - cpr_annual) ** (1 / 12)
monthly_cdr = 1 - (1 - cdr_annual) ** (1 / 12)

# Calculate monthly loan payment


def monthly_payment(P, r, n):
    return P * r * (1 + r)**n / ((1 + r)**n - 1)


monthly_loan_payment = monthly_payment(
    avg_loan_size, loan_monthly_rate, loan_term_months)
monthly_pool_payment = monthly_loan_payment * num_loans

# Initialize tracking variables
pool_balance = pool_start_balance + \
    overcollateralization  # Include overcollateralization
senior_balance = senior_start_balance
mezzanine_balance = mezzanine_start_balance
subordinate_balance = subordinate_start_balance
equity_balance = equity_start_balance
cashflow_data = []

# Cash flow calculation
for month in range(1, loan_term_months + 1):
    if pool_balance <= 0:
        print(f'The pool balance at month {month} turns to zero')
        break

    print(f'The pool balance at month {month} equals to {pool_balance:,.0f}')
    print(
        f'The senior balance at month {month} equals to {senior_balance:,.0f}')
    print(
        f'The mezzanine balance at month {month} equals to {mezzanine_balance:,.0f}')
    print(
        f'The subordinate balance at month {month} equals to {subordinate_balance:,.0f}')
    print(
        f'The equity balance at month {month} equals to {equity_balance:,.0f}')

    # Step 1: Interest and scheduled principal
    interest_payment = pool_balance * loan_monthly_rate
    print(
        f'The interest payment at month {month} equals to {interest_payment:,.0f}')
    scheduled_principal = monthly_pool_payment - interest_payment
    print(
        f'The scheduled principal at month {month} equals to {scheduled_principal:,.0f}')

    # Step 2: Prepayment
    remaining_after_sched = pool_balance - scheduled_principal
    print(
        f'The remaining pool balance after scheduled principal at month {month} equals to {remaining_after_sched:,.0f}')
    prepayment = remaining_after_sched * monthly_cpr
    print(f'The prepayment at month {month} equals to {prepayment:,.0f}')

    # Step 3: Default
    remaining_after_prepay = remaining_after_sched - prepayment
    print(
        f'The remaining pool balance after prepayment at month {month} equals to {remaining_after_prepay:,.0f}')
    default_amt = remaining_after_prepay * monthly_cdr
    print(f'The default amount at month {month} equals to {default_amt:,.0f}')
    loss_amt = default_amt * loss_severity
    print(
        f'The default loss amount at month {month} equals to {loss_amt:,.0f}')

    # Step 4: Principal distribution
    total_principal_avail = max(
        0, scheduled_principal + prepayment - default_amt)
    print(
        f'The aggregate principal at month {month} equals to {total_principal_avail:,.0f}')

    # Step 5: Loss absorption (equity -> subordinate -> mezzanine -> senior)
    absorbed_loss = 0
    if equity_balance > 0:
        absorbed_loss = min(loss_amt, equity_balance)
        print(
            f'The loss absorbed by equity at month {month} equals to {absorbed_loss:,.0f}')
        equity_balance -= absorbed_loss
        print(
            f'The residual equity balance at month {month} equals to {equity_balance:,.0f}')
    elif subordinate_balance > 0:
        remaining_loss = loss_amt - absorbed_loss
        absorbed_loss += min(remaining_loss, subordinate_balance)
        subordinate_balance -= min(remaining_loss, subordinate_balance)
        print(
            f'The loss absorbed by subordinate at month {month} equals to {absorbed_loss:,.0f}')
        print(
            f'The residual subordinate balance at month {month} equals to {subordinate_balance:,.0f}')
    elif mezzanine_balance > 0:
        remaining_loss = loss_amt - absorbed_loss
        absorbed_loss += min(remaining_loss, mezzanine_balance)
        mezzanine_balance -= min(remaining_loss, mezzanine_balance)
        print(
            f'The loss absorbed by mezzanine at month {month} equals to {absorbed_loss:,.0f}')
        print(
            f'The residual mezzanine balance at month {month} equals to {mezzanine_balance:,.0f}')

    # Step 6: Servicing fee
    servicing_fee_monthly = pool_balance * (servicing_fee / 12)
    print(
        f'The servicing fee at month {month} equals to {servicing_fee_monthly:,.0f}')

    # Step 7: Tranche cash flows (sequential pay structure)
    # Senior tranche
    senior_interest = senior_balance * senior_monthly_coupon
    print(
        f'The interest received for senior tranche at month {month} equals to {senior_interest:,.0f}')
    senior_principal = min(total_principal_avail, senior_balance)
    print(
        f'The principal received for senior tranche at month {month} equals to {senior_principal:,.0f}')
    senior_cash_flow = senior_interest + senior_principal
    print(
        f'The cash flow of senior tranche at month {month} equals to {senior_cash_flow:,.0f}')

    # Mezzanine tranche
    remaining_principal = max(0, total_principal_avail - senior_principal)
    print(
        f'The principal left for mezzanine tranche at month {month} equals to {remaining_principal:,.0f}')
    mezzanine_interest = mezzanine_balance * mezzanine_monthly_coupon
    print(
        f'The interest received for mezzanine tranche at month {month} equals to {mezzanine_interest:,.0f}')
    mezzanine_principal = min(remaining_principal, mezzanine_balance)
    print(
        f'The principal received for mezzanine tranche at month {month} equals to {mezzanine_principal:,.0f}')
    mezzanine_cash_flow = mezzanine_interest + mezzanine_principal
    print(
        f'The cash flow of mezzanine tranche at month {month} equals to {mezzanine_cash_flow:,.0f}')

    # Subordinate tranche
    remaining_principal = max(0, remaining_principal - mezzanine_principal)
    print(
        f'The principal left for subordinate tranche at month {month} equals to {remaining_principal:,.0f}')
    subordinate_interest = subordinate_balance * subordinate_monthly_coupon
    print(
        f'The interest received for subordinate tranche at month {month} equals to {subordinate_interest:,.0f}')
    subordinate_principal = min(remaining_principal, subordinate_balance)
    print(
        f'The principal received for subordinate tranche at month {month} equals to {subordinate_principal:,.0f}')
    subordinate_cash_flow = subordinate_interest + subordinate_principal
    print(
        f'The cash flow of subordinate tranche at month {month} equals to {subordinate_cash_flow:,.0f}')

    # Equity tranche
    remaining_principal = max(0, remaining_principal - subordinate_principal)
    print(
        f'The principal left for equity tranche at month {month} equals to {remaining_principal:,.0f}')
    equity_interest = equity_balance * equity_monthly_coupon
    print(
        f'The interest received for equity tranche at month {month} equals to {equity_interest:,.0f}')
    equity_principal = min(remaining_principal, equity_balance)
    print(
        f'The principal received for equity tranche at month {month} equals to {equity_principal:,.0f}')
    equity_cash_flow = equity_interest + equity_principal
    print(
        f'The cash flow of equity tranche at month {month} equals to {equity_cash_flow:,.0f}')

    # Update balances
    senior_balance -= senior_principal
    mezzanine_balance -= mezzanine_principal
    subordinate_balance -= subordinate_principal
    equity_balance -= equity_principal
    pool_balance -= (scheduled_principal + prepayment + default_amt)

    # Store data
    cashflow_data.append({
        "Month": month,
        "Pool_Balance": pool_balance,
        "Senior_Balance": senior_balance,
        "Mezzanine_Balance": mezzanine_balance,
        "Subordinate_Balance": subordinate_balance,
        "Equity_Balance": equity_balance,
        "Scheduled_Principal": scheduled_principal,
        "Prepayment": prepayment,
        "Default_Amount": default_amt,
        "Loss_Absorbed": absorbed_loss,
        "Principal_Distributed": total_principal_avail,
        "Servicing_Fee": servicing_fee_monthly,
        "Senior_Interest": senior_interest,
        "Senior_Principal": senior_principal,
        "Senior_Cash_Flow": senior_cash_flow,
        "Mezzanine_Interest": mezzanine_interest,
        "Mezzanine_Principal": mezzanine_principal,
        "Mezzanine_Cash_Flow": mezzanine_cash_flow,
        "Subordinate_Interest": subordinate_interest,
        "Subordinate_Principal": subordinate_principal,
        "Subordinate_Cash_Flow": subordinate_cash_flow,
        "Equity_Interest": equity_interest,
        "Equity_Principal": equity_principal,
        "Equity_Cash_Flow": equity_cash_flow
    })

# Convert to DataFrame
df_cashflow = pd.DataFrame(cashflow_data)

# Assuming df_cashflow is available from previous code
# Plotting Interest, Principal, and Cash Flow for each tranche over 60 months
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle("ABS Cash Flows by Tranche (60 Months)", fontsize=16)

# Senior Tranche Plots
axes[0, 0].plot(df_cashflow["Month"],
                df_cashflow["Senior_Interest"], color='blue')
axes[0, 0].set_title("Senior Interest")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Amount ($)")
axes[0, 0].set_xlim(0, 60)
axes[0, 0].set_xticks(np.arange(0, 61, 12))  # Ticks every year
axes[0, 0].grid(True)

axes[0, 1].plot(df_cashflow["Month"],
                df_cashflow["Senior_Principal"], color='green')
axes[0, 1].set_title("Senior Principal")
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Amount ($)")
axes[0, 1].set_xlim(0, 60)
axes[0, 1].set_xticks(np.arange(0, 61, 12))
axes[0, 1].grid(True)

axes[0, 2].plot(df_cashflow["Month"],
                df_cashflow["Senior_Cash_Flow"], color='purple')
axes[0, 2].set_title("Senior Cash Flow")
axes[0, 2].set_xlabel("Month")
axes[0, 2].set_ylabel("Amount ($)")
axes[0, 2].set_xlim(0, 60)
axes[0, 2].set_xticks(np.arange(0, 61, 12))
axes[0, 2].grid(True)

# Mezzanine Tranche Plots
axes[1, 0].plot(df_cashflow["Month"],
                df_cashflow["Mezzanine_Interest"], color='blue')
axes[1, 0].set_title("Mezzanine Interest")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Amount ($)")
axes[1, 0].set_xlim(0, 60)
axes[1, 0].set_xticks(np.arange(0, 61, 12))
axes[1, 0].grid(True)

axes[1, 1].plot(df_cashflow["Month"],
                df_cashflow["Mezzanine_Principal"], color='green')
axes[1, 1].set_title("Mezzanine Principal")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Amount ($)")
axes[1, 1].set_xlim(0, 60)
axes[1, 1].set_xticks(np.arange(0, 61, 12))
axes[1, 1].grid(True)

axes[1, 2].plot(df_cashflow["Month"],
                df_cashflow["Mezzanine_Cash_Flow"], color='purple')
axes[1, 2].set_title("Mezzanine Cash Flow")
axes[1, 2].set_xlabel("Month")
axes[1, 2].set_ylabel("Amount ($)")
axes[1, 2].set_xlim(0, 60)
axes[1, 2].set_xticks(np.arange(0, 61, 12))
axes[1, 2].grid(True)

# Equity Tranche Plots
axes[2, 0].plot(df_cashflow["Month"],
                df_cashflow["Equity_Interest"], color='blue')
axes[2, 0].set_title("Equity Interest")
axes[2, 0].set_xlabel("Month")
axes[2, 0].set_ylabel("Amount ($)")
axes[2, 0].set_xlim(0, 60)
axes[2, 0].set_xticks(np.arange(0, 61, 12))
axes[2, 0].grid(True)

axes[2, 1].plot(df_cashflow["Month"],
                df_cashflow["Equity_Principal"], color='green')
axes[2, 1].set_title("Equity Principal")
axes[2, 1].set_xlabel("Month")
axes[2, 1].set_ylabel("Amount ($)")
axes[2, 1].set_xlim(0, 60)
axes[2, 1].set_xticks(np.arange(0, 61, 12))
axes[2, 1].grid(True)

axes[2, 2].plot(df_cashflow["Month"],
                df_cashflow["Equity_Cash_Flow"], color='purple')
axes[2, 2].set_title("Equity Cash Flow")
axes[2, 2].set_xlabel("Month")
axes[2, 2].set_ylabel("Amount ($)")
axes[2, 2].set_xlim(0, 60)
axes[2, 2].set_xticks(np.arange(0, 61, 12))
axes[2, 2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("abs_tranche_cashflows_60_months.png")

# Save to CSV for portability
df_cashflow.to_csv("abs_cashflows.csv", index=False)
