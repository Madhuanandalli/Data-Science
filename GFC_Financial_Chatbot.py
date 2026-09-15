{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "2c785a73-2f87-4440-9a8c-19f1e34ff0a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Welcome to the GFC Financial Chatbot!\n",
      "You can ask the following predefined questions:\n",
      "1. What was Microsoft's revenue in 2025?\n",
      "2. What was Apple's net income in 2025?\n",
      "3. What was Tesla's operating cash flow in 2025?\n",
      "4. What was Microsoft's revenue growth in 2025?\n",
      "5. Which company had the highest net profit margin in 2025?\n",
      "Type 'exit' to end the conversation.\n",
      "\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  what was microsoft's revenue in 2025?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chatbot: Microsoft's revenue in 2025 was $281,724 million.\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  what was apple's net income in 2025?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chatbot: Apple's net income in 2025 was $112,010 million.\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  what was tesla's operating cash flow in 2025?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chatbot: Tesla's operating cash flow in 2025 was $14,747 million.\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  what was microsoft's revenue growth in 2025?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chatbot: Microsoft's revenue grew by 14.93% from 2024 to 2025.\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  which company had the highest net profit margin in 2025?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chatbot: Microsoft had the highest net profit margin in 2025 at 36.15%.\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  what is Apple's stock price tomorrow?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chatbot: Sorry, I can only provide information on predefined financial queries. You can ask about revenue, net income, operating cash flow, revenue growth, or net profit margin.\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  exit\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chatbot: Thank you for using the GFC Financial Chatbot. Goodbye!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load financial data\n",
    "df = pd.read_excel(\"GFC_10K_Financial_Data_2023_2025.xlsx\")\n",
    "\n",
    "# Sort data\n",
    "df = df.sort_values([\"Company\", \"Fiscal Year\"])\n",
    "\n",
    "# Calculate required metrics\n",
    "df[\"Revenue Growth (%)\"] = (\n",
    "    df.groupby(\"Company\")[\"Total Revenue\"].pct_change() * 100\n",
    ")\n",
    "\n",
    "df[\"Net Profit Margin (%)\"] = (\n",
    "    df[\"Net Income\"] / df[\"Total Revenue\"] * 100\n",
    ")\n",
    "\n",
    "\n",
    "def simple_chatbot(user_query):\n",
    "    user_query = user_query.strip().lower()\n",
    "\n",
    "    # Query 1: Microsoft's revenue in 2025\n",
    "    if user_query == \"what was microsoft's revenue in 2025?\":\n",
    "        revenue = df[\n",
    "            (df[\"Company\"] == \"Microsoft\") &\n",
    "            (df[\"Fiscal Year\"] == 2025)\n",
    "        ][\"Total Revenue\"].iloc[0]\n",
    "\n",
    "        return f\"Microsoft's revenue in 2025 was ${revenue:,.0f} million.\"\n",
    "\n",
    "    # Query 2: Apple's net income in 2025\n",
    "    elif user_query == \"what was apple's net income in 2025?\":\n",
    "        net_income = df[\n",
    "            (df[\"Company\"] == \"Apple\") &\n",
    "            (df[\"Fiscal Year\"] == 2025)\n",
    "        ][\"Net Income\"].iloc[0]\n",
    "\n",
    "        return f\"Apple's net income in 2025 was ${net_income:,.0f} million.\"\n",
    "\n",
    "    # Query 3: Tesla's operating cash flow in 2025\n",
    "    elif user_query == \"what was tesla's operating cash flow in 2025?\":\n",
    "        cash_flow = df[\n",
    "            (df[\"Company\"] == \"Tesla\") &\n",
    "            (df[\"Fiscal Year\"] == 2025)\n",
    "        ][\"Operating Cash Flow\"].iloc[0]\n",
    "\n",
    "        return f\"Tesla's operating cash flow in 2025 was ${cash_flow:,.0f} million.\"\n",
    "\n",
    "    # Query 4: Microsoft's revenue growth in 2025\n",
    "    elif user_query == \"what was microsoft's revenue growth in 2025?\":\n",
    "        growth = df[\n",
    "            (df[\"Company\"] == \"Microsoft\") &\n",
    "            (df[\"Fiscal Year\"] == 2025)\n",
    "        ][\"Revenue Growth (%)\"].iloc[0]\n",
    "\n",
    "        return f\"Microsoft's revenue grew by {growth:.2f}% from 2024 to 2025.\"\n",
    "\n",
    "    # Query 5: Highest net profit margin in 2025\n",
    "    elif user_query == \"which company had the highest net profit margin in 2025?\":\n",
    "        data_2025 = df[df[\"Fiscal Year\"] == 2025]\n",
    "\n",
    "        highest = data_2025.loc[\n",
    "            data_2025[\"Net Profit Margin (%)\"].idxmax()\n",
    "        ]\n",
    "\n",
    "        return (\n",
    "            f\"{highest['Company']} had the highest net profit margin \"\n",
    "            f\"in 2025 at {highest['Net Profit Margin (%)']:.2f}%.\"\n",
    "        )\n",
    "\n",
    "    # Error handling\n",
    "    else:\n",
    "        return (\n",
    "            \"Sorry, I can only provide information on predefined financial \"\n",
    "            \"queries. You can ask about revenue, net income, operating cash \"\n",
    "            \"flow, revenue growth, or net profit margin.\"\n",
    "        )\n",
    "\n",
    "\n",
    "# Start chatbot\n",
    "print(\"Welcome to the GFC Financial Chatbot!\")\n",
    "print(\"You can ask the following predefined questions:\")\n",
    "print(\"1. What was Microsoft's revenue in 2025?\")\n",
    "print(\"2. What was Apple's net income in 2025?\")\n",
    "print(\"3. What was Tesla's operating cash flow in 2025?\")\n",
    "print(\"4. What was Microsoft's revenue growth in 2025?\")\n",
    "print(\"5. Which company had the highest net profit margin in 2025?\")\n",
    "print(\"Type 'exit' to end the conversation.\\n\")\n",
    "\n",
    "\n",
    "while True:\n",
    "    user_query = input(\"You: \")\n",
    "\n",
    "    if user_query.lower().strip() == \"exit\":\n",
    "        print(\"Chatbot: Thank you for using the GFC Financial Chatbot. Goodbye!\")\n",
    "        break\n",
    "\n",
    "    response = simple_chatbot(user_query)\n",
    "    print(\"Chatbot:\", response)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de02fa42-a9ed-4fd3-9c8c-38c5c3a537a3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ae5ec6c-593f-4b41-8e50-7a5125eb5bf2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
