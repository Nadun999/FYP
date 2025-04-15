from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.schema import OutputParserException
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
import mysql.connector
import inference.settings as settings
from pydantic import BaseModel, Field
from textwrap import dedent

# configuration for handling callbacks and LLM setup
langfuse_handler = CallbackHandler(
    secret_key="sk-lf-1b6737ff-b2f5-4468-a0ff-31a2f8c89592",
    public_key="pk-lf-5b8579d9-c0a0-4126-8fe1-c748668b8a77",
    host="https://us.cloud.langfuse.com", 
)

def parse_response(output, parser, llm=ChatOpenAI(model="gpt-4o", openai_api_key="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")):
    try:
        parsed = parser.parse(output)
    except OutputParserException:
        try:
            new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            parsed = new_parser.parse(output.content)
            print("Fixed parsing errors.")
        except Exception as e:
            print("Failed to fix parsing errors.")
            parsed = None  
    return parsed


def llm_invoke(llm, messages):
    return llm.invoke(messages.to_messages(), {"callbacks": [langfuse_handler]})

class SQLQueryModel(BaseModel):
    sql_query: str = Field(
        description="SQL query generated based on the user's natural language input."
    )

def generate_sql_query(user_input, llm):
    """
    This function generates a SQL query based on the user's input using GPT-4 and the relationships in the cricket database.
    """
    # define the parser for SQL output
    parser = PydanticOutputParser(pydantic_object=SQLQueryModel)
    format_instructions = parser.get_format_instructions()

    prompt_template = dedent(
        """
        You are an AI assistant specialized in generating SQL queries from multiple conditions and tables from natural language.

        The database consists of the following tables and relationships:
        - `batting`: Contains information about players' batting performance.
          - Columns: `Runs` (runs scored), `Mins` (minutes batted), `BF` (balls faced), `4s`, `6s`, `SR` (strike rate), `Pos` (batting position), `Dismissal` (how the batsman got out), `Inns` (1 for 1st innings, 2 for 2nd innings), `Opposition`, `Ground`, `Start Date`, `t20_match_no`, `player_id`, `ground_id`, `opposition_country_id`, `home_away`, `match_result_id`, `in_finals_id`.
          - Foreign Keys: `player_id` references `player(player_id)`, `ground_id` references `ground(ground_id)`,
            `opposition_country_id` references `country(country_id)`, `match_result_id` references `match_result(match_result_id)`,
            `in_finals_id` references `in_finals(match_stage_id)`, `t20_match_no` to identify specific matches.
        - `bowling`: Contains information about players' bowling performance.
          - Columns: `Overs` (number of overs bowled), `Mdns` (number of maidens bowled), `Runs` (runs conceded), `Wkts` (wickets taken), `Econ` (economy rate), `Pos` (bowling position), `Inns` (1 for 1st innings, 2 for 2nd innings), `t20_match_no`, `player_id`, `match_result_id`, `home_away`, `in_finals_id`, `start_date`.
          - Foreign Keys: `player_id` references `player(player_id)`, `ground_id` references `ground(ground_id)`,
            `opposition_country_id` references `country(country_id)`, `match_result_id` references `match_result(match_result_id)`,
            `in_finals_id` references `in_finals(match_stage_id)`, `t20_match_no` to identify specific matches.
        - `fielding`: Contains information about players' fielding performance.
          - Columns: `Dis` (number of dismissals), `Ct` (catches taken), `St` (stumpings), `Ct Wk` (catches as wicketkeeper), `Ct Fi` (catches as a fielder), `Inns` (1 for 1st innings, 2 for 2nd innings), `t20_match_no`, `player_id`, `match_result_id`, `home_away`, `start_date`.
          - Foreign Keys: `player_id` references `player(player_id)`, `ground_id` references `ground(ground_id)`,
            `opposition_country_id` references `country(country_id)`, `match_result_id` references `match_result(match_result_id)`,
            `in_finals_id` references `in_finals(match_stage_id)`, `t20_match_no` to identify specific matches.
        - `player`: Stores details about players (`player_id` as primary key).
          - Columns: `player_id`, `player_name`, `country_id` (references `country(country_id)` to identify which country the player belongs to).
        - `country`: Stores information about countries (`country_id` as primary key).
          - Columns: `country_id`, `country_name` (e.g., 'India', 'Australia').
        - `ground`: Stores information about match grounds (`ground_id` as primary key).
        - `match_result`: Stores match results (`match_result_id` as primary key).
          - Columns: `match_result_id`, `match_result` (if `match_result_id`is 1 then 'match_result' is 'won match', `match_result_id`is 2 then 'match_result' is 'lost match', `match_result_id`is 3 then 'match_result' is 'tied match', `match_result_id`is 4 then 'match_result' is 'no result').
        - `in_finals`: Stores information on whether a match was a final (`match_stage_id` as primary key).

        When generating the SQL query, consider the following:
        - The query is properly formatted and valid for MySQL.
        - The response should be in JSON format with a single key named `sql_query`.
        - Example JSON response: {{ "sql_query": "SELECT * FROM ...;" }}
        - Use `CAST()` for type conversions instead of `::`.
        - Avoid any syntax that is specific to PostgreSQL or other SQL variants.
        - Handle multiple conditions, such as filtering by runs, wickets, and specific years.
        - Use Common Table Expressions (CTEs) or subqueries if needed to calculate aggregates or percentages.
        - Use `DISTINCT` when counting or selecting matches to avoid duplication.
        - Use aggregate functions such as `COUNT()` to calculate total counts.
        - Make sure to use `GROUP BY` where required when using aggregate functions in select statements.
        - Use `NOT IN` subqueries to filter out matches involving certain players when requested.
        - Return only the columns that are necessary to answer the user's question.
        - If you encounter a player name or any entity that is misspelled, use fuzzy matching or approximate search techniques to identify the closest match.
        - To determine match outcomes involving a specific player (e.g., Arshdeep Singh), join `player` and `match_result` tables to verify the country and match outcome respectively.
        - Use `t20_match_no` as the unique identifier for each match, linking only batting, bowling, and fielding.
        - Focus on matches where a specific player meets the conditions given in the user's question.
        - Use a step-by-step approach with Common Table Expressions (CTEs) to break down the query into logical steps:
            1. First, identify matches where specific conditions are met (e.g., a player took a certain number of wickets).
            2. Then, determine how many of these matches meet another condition (e.g., matches were won by India).
            3. Finally, calculate any required performance metrics or counts based on these results.    
        - Use the `home_away` column from `batting`, `bowling`, or `fielding` to determine if the match was played at home, away, or neutral.
        - When determining the number of matches played, consider using joins across multiple tables (`batting`, `bowling`, etc.) and apply `UNION` if needed to avoid double counting.
        - Ensure that the conditions for finals, such as checking `in_finals_id`, are correctly applied to identify final matches.
        - Include additional details like `start_date` and `opposition_country_id` to provide more context for match results, such as the opponent team and match date.
        - When calculating performance metrics (e.g., economy rate, win percentage when conditions are met), use CTEs or subqueries to first identify relevant matches and then apply aggregate functions to derive the final output.



        Format your response as follows:
        {format_instructions}

        Given the user's question, generate an appropriate SQL query that joins the necessary tables based on the foreign key relationships.

        User Input:
        {user_input}
        """
    )

    prompt = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(prompt_template)],
        input_variables=["user_input", "format_instructions"],
    )

    # format the prompt with user input and format instructions
    _input = prompt.format_prompt(
        user_input=user_input, format_instructions=format_instructions
    )
    
    # invoke the LLM to generate the SQL query
    output = llm_invoke(llm, _input)
    
    print("Raw Output from LLM:", output.content)

    # parse the output to extract the SQL query
    parsed = parse_response(output.content, parser)
    if parsed is None:
        raise ValueError("Failed to parse the response. Please check the output.")

    sql_query = parsed.sql_query

    # initialize result to None to ensure it is defined
    result = None

    # execute the SQL query against MySQL database
    try:
        connection = mysql.connector.connect(
            host=settings.DB_HOST,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME,
            port=settings.DB_PORT
        )
        cursor = connection.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        for row in result:
            print(row)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return result

# if __name__ == "__main__":
#     user_question = "In how many matches Arshdeep singh took 2 wickets and scored 2 runs?" 
#     llm = ChatOpenAI(model="gpt-4o", openai_api_key="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")
#     result = generate_sql_query(user_question, llm)
#     print("Query Result:", result)


player_stats = {
    "Virat_Kohli": {
        "Span": "2010-2024",
        "Mat": 125,
        "Runs": 4188,
        "HS": "122*",
        "Bat Avg": 48.69,
        "SR": 137.04,
        "100s": 1,
        "50s": 38,
        "Wkts": 4,
        "Econ": 8.30,
        "BBI": "1/13",
        "Bowl Avg": 51.00,
        "5w": 0,
        "Ct": 54
    },
    "Arshdeep_Singh": {
        "Span": "2022-2025",
        "Mat": 63,
        "Runs": 71,
        "HS": 12,
        "Bat Avg": 8.87,
        "SR": 116.39,
        "100s": 0,
        "50s": 0,
        "Wkts": 99,
        "Econ": 8.29,
        "BBI": "4/9",
        "Bowl Avg": 18.30,
        "5w": 0,
        "Ct": 16
    },
    "Axar_Patel": {
        "Span": "2015-2025",
        "Mat": 66,
        "Runs": 498,
        "HS": 65,
        "Bat Avg": 19.92,
        "SR": 139.32,
        "100s": 0,
        "50s": 1,
        "Wkts": 71,
        "Econ": 7.30,
        "BBI": "3/9",
        "Bowl Avg": 22.12,
        "5w": 0,
        "Ct": 25
    },
    "Jasprit_Bumrah": {
        "Span": "2016-2024",
        "Mat": 70,
        "Runs": 8,
        "HS": 7,
        "Bat Avg": 2.66,
        "SR": 57.14,
        "100s": 0,
        "50s": 0,
        "Wkts": 89,
        "Econ": 6.27,
        "BBI": "3/7",
        "Bowl Avg": 17.74,
        "5w": 0,
        "Ct": 9
    },
    "Kuldeep_Yadav": {
        "Span": "2017-2024",
        "Mat": 40,
        "Runs": 46,
        "HS": "23*",
        "Bat Avg": 11.50,
        "SR": 77.96,
        "100s": 0,
        "50s": 0,
        "Wkts": 69,
        "Econ": 6.77,
        "BBI": "5/17",
        "Bowl Avg": 14.07,
        "5w": 2,
        "Ct": 12
    },
    "Ravindra_Jadeja": {
        "Span": "2009-2024",
        "Mat": 74,
        "Runs": 515,
        "HS": "46*",
        "Bat Avg": 21.45,
        "SR": 127.16,
        "100s": 0,
        "50s": 0,
        "Wkts": 54,
        "Econ": 7.13,
        "BBI": "3/15",
        "Bowl Avg": 29.85,
        "5w": 0,
        "Ct": 28
    }
}