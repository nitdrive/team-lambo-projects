class WebsiteLookUpByFileName:
    @classmethod
    def get_websites_by_file_name(cls, file_name):
        websites_associated = {
            'About-Vanguard.html': 'https://en.wikipedia.org/wiki/The_Vanguard_Group',
            'Robo-Advisor_-_Automated_Investing_Services___Vanguard.html': 'https://investor.vanguard.com/advice/robo-advisor',
            '529_plan_for_college_savings___Vanguard.html': 'https://investor.vanguard.com/accounts-plans/529-plans',
            'Invest_for_your_future_with_Vanguard_IRAs___Vanguard.html': 'https://investor.vanguard.com/accounts-plans/iras',
            'Investment_accounts__plans_for_all_investment_goals___Vanguard.html': 'https://investor.vanguard.com/accounts-plans',
            'Organization_Accounts___Vanguard.html': 'https://investor.vanguard.com/accounts-plans/organization-accounts',
            'Roth_IRA__What_is_a_Roth_IRA____Vanguard.html': 'https://investor.vanguard.com/accounts-plans/iras/roth-ira',
            'Small_business_retirement_plans___Vanguard.html': 'https://investor.vanguard.com/accounts-plans/small-business-retirement-plans',
            'Trust_Account__What_Is_It_and_How_To_Get_Started___Vanguard.html': 'https://investor.vanguard.com/accounts-plans/trust-accounts',
            'Vanguard_Cash_Plus_Account___Vanguard.html': 'https://investor.vanguard.com/accounts-plans/vanguard-cash-plus-account',
            'What_is_a_Brokerage_account_and_how_does_it_work____Vanguard.html': 'https://investor.vanguard.com/accounts-plans/brokerage-accounts',
            'How_to_invest_to_reach_your_financial_goals__Vanguard.html': 'https://investor.vanguard.com/investor-resources-education/how-to-invest'
        }

        if file_name in websites_associated:
            return websites_associated[file_name]

        return file_name
