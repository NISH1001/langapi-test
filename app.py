#!/usr/bin/env python3

import copy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import time
from datetime import datetime

from typing import Tuple, Optional

from tqdm import tqdm

import json
import requests
from urllib.parse import urljoin
import posixpath
import io

MAP_KEY = os.getenv("MAP_KEY")
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
SOURCE = "VIIRS_SNPP_NRT"
FIG_SIZE = (15, 10)


class DialogParser:
    PARSER_PROMPT = f"""
        You are a tool that converts incoming natural text to list of parameters that can be used for FIRMS (Fire Information for Resource Management System) API.
        There are 3 parameter values (named entitites) that need to be understood and parsed from the text.

        Parameters:
            Area: This is the bbox area of geojson. It has 4 integer values in the format <WEST>,<SOUTH>,<EAST>,<NORTH>. If not provided, the area will be entire world.
            Date: This is the date in the format YYYY-MM-DD for which forest data has to be fetched. This can be a single value or list of data based on user input texts.
            Day Range: This is the number of days for which data is fetched. Its value strictly ranges from 1 to 10.


        Your task is to parse input texts and give the values of all these 3 parameters in the provided format.

        Following instructions are used to show how to parse date values:
            Assume today's date is {time.strftime('%Y-%m-%d')}.
            If user specifies exactly N years ago, use today's date and compute the date as today's date - N and provide single value date. Don't generate list in this case. Only provide single date.
            If date is not provided directly, you can get relative date based on today's date. For example, if the users says 'data from last year', you have to use today's date and give the date of last year. And generate list of YYYY-MM-DD values starting from January of that year to ending at December of that year.
            If user aks for data trend for age range, provide the date as list starting from January of starting year and ending to December of end year. If end year is not provided, assume end year from today's date.
            If user specifies a specific month and year, just use first day of that month and only that year and give single value date. Keep the day range value as it is.
            If user specifies trend since certain year, generate list of dates from that year to today.


        If area is not provided, output "world" as value for area. Strictly provide area bbox as geojson. If area is not geosjon, fetch geojson bbox value as mentioned format of <WEST>,<SOUTH>,<EAST>,<NORTH>.

        If day range is not provided, default to 1. Only output the value in range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. If value is greater than 10, threshold it to 10.

        Strictly output json only. Remove any additional notes and comments.
    """
    RESTRUCTURE_PROMPT = f"""
        Give me the data strictly in json format as:
            'area': <area>
            'date': <date>
            'range': <range>

        Don't output anything else. Make sure keys and values are in double quotes. Not single quotes.
        Remove any additional notes and comments. Don't add any extra comments and texts anywhere.

        If date range is greater than today's date which is {time.strftime('%Y-%m-%d')}, set the date to today's date.

        Area should strictly be geojson for given location. If value for area is not geojson, fetch geojson and provide that. Output the area value in the format <WEST>,<SOUTH>,<EAST>,<NORTH>.

    """

    def __init__(
        self,
        llm,
        parser_prompt: Optional[str] = None,
        restructure_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.parser_prompt = parser_prompt or DialogParser.PARSER_PROMPT
        self.restructure_prompt = restructure_prompt = DialogParser.RESTRUCTURE_PROMPT

    @property
    def today_str(self) -> str:
        return time.strftime("%Y-%m-%d")

    def run_model(self, messages: Tuple[SystemMessage, HumanMessage]) -> str:
        output = self.llm._generate(messages)

        # Extract and return the generated text from the model output
        return output.generations[0].text.strip()

    def parse(self, text) -> dict:
        prompt_message = SystemMessage(content=self.parser_prompt.strip())
        user_message = HumanMessage(content=text.strip())
        restructure_message = SystemMessage(content=self.restructure_prompt.strip())

        message = (prompt_message, user_message, restructure_message)

        res = self.run_model(message)
        print(res)
        return json.loads(res)


def call_api(
    base_url, api_key, area, date, day_range, source="VIIRS_SNPP_NRT"
) -> pd.DataFrame:
    url = posixpath.join(
        f"{api_key}", f"{source}", f"{area or 'world'}", f"{day_range}", f"{date}"
    )
    url = urljoin(base_url, url)
    print(f"url = {url}")
    response = requests.get(url).text
    return pd.read_csv(io.StringIO(response), sep=",")


def plot_brightness(data, fig_size=FIG_SIZE, fig=None):
    """
    Plot ti4 and ti5 time series
    """
    if fig is None:
        fig = plt.figure(figsize=FIG_SIZE)
    ax = sns.lineplot(
        data=data,
        x="acq_date",
        y="bright_ti4",
        label="bright_ti4",
    )
    ax = sns.lineplot(
        data=data,
        x="acq_date",
        y="bright_ti5",
        label="bright_ti5",
    )
    ax.set_title("brightness trend", fontsize=15)
    ax.set_ylabel("bright_ti4/5")
    ax.tick_params(axis="x", rotation=90)
    plt.legend()
    plt.show()
    return fig


def plot_ti4(data, fig_size=FIG_SIZE, fig=None):
    """
    Plot ti4 w.r.t day and night
    """
    if fig is None:
        fig = plt.figure(figsize=FIG_SIZE)
    ax = sns.lineplot(data=data, x="acq_date", y="bright_ti4", hue="daynight")
    ax.set_title("ti4 trend across day and night", fontsize=15)
    ax.tick_params(axis="x", rotation=90)
    plt.show()
    return fig


def trend_pass(parser, text):
    params = parser.parse(text)
    params["range"] = max(1, min(params.get("range", 1), 10))
    print(params)

    dates = params.get("date", time.strftime("time.strftime('%Y-%m-%d')"))

    dates = dates if isinstance(dates, list) else [dates]

    data_agg = []
    fig = None
    for date in tqdm(dates):
        print(f"date = {date}")
        data = call_api(
            base_url=BASE_URL,
            api_key=MAP_KEY,
            source=SOURCE,
            area=params.get("area", "world"),
            date=date,
            day_range=params.get("range", 1),
        )
        if data.empty:
            continue

        data_agg.append(data)

        data = pd.concat(data_agg)
        data = data.reset_index(drop=True)

        fig = plot_brightness(data)
        fig.canvas.draw()

        fig = plot_ti4(data)
        fig.canvas.draw()

        fig.canvas.flush_events()
    return data


def trend_pass_streamlit(params) -> pd.DataFrame:
    params = copy.deepcopy(params)
    params["range"] = max(1, min(params.get("range", 1), 10))
    print(params)

    dates = params.get("date", time.strftime("time.strftime('%Y-%m-%d')"))

    dates = dates if isinstance(dates, list) else [dates]

    data_agg = []
    for date in tqdm(dates):
        print(f"date = {date}")
        data = call_api(
            base_url=BASE_URL,
            api_key=MAP_KEY,
            source=SOURCE,
            area=params.get("area", "world"),
            date=date,
            day_range=params.get("range", 1),
        )
        if data.empty:
            continue

        data_agg.append(data)

    return pd.concat(data_agg).reset_index() if data_agg else pd.DataFrame()


def debug():
    st.title("trend analysis demo - NLP")

    data = pd.concat(
        [
            call_api(
                base_url=BASE_URL,
                api_key=MAP_KEY,
                source=SOURCE,
                area="-88.473227,30.137988,-84.888246,35.008028",
                date="2023-01-01",
                day_range=5,
            ),
            call_api(
                base_url=BASE_URL,
                api_key=MAP_KEY,
                source=SOURCE,
                area="-88.473227,30.137988,-84.888246,35.008028",
                date="2023-02-01",
                day_range=1,
            ),
            call_api(
                base_url=BASE_URL,
                api_key=MAP_KEY,
                source=SOURCE,
                area="-88.473227,30.137988,-84.888246,35.008028",
                date="2023-03-01",
                day_range=5,
            ),
        ]
    )

    viz_df = (
        data.groupby("acq_date")[["bright_ti4", "bright_ti5"]]
        .mean()
        .reset_index()
        .rename(columns={"acq_date": "index"})
        .set_index("index")
    )
    st.subheader("Average brightness across time")
    st.line_chart(data=viz_df)

    viz_df = (
        data.groupby("acq_date")["confidence"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"acq_date": "index"})
        .set_index("index")
    )
    st.subheader("Confidence level across time")
    st.bar_chart(data=viz_df)


def run_streamlit(parser):
    st.title("trend analysis demo - NLP")

    text = st.text_input("text", "")
    if text:
        with st.spinner(text="Parsing input..."):
            params = parser.parse(text)

        with st.spinner(text="Aggregating data..."):
            data = trend_pass_streamlit(params)

        st.success("Done!")

        if not data.empty:
            print(data)
            viz_df = (
                data.groupby("acq_date")[["bright_ti4", "bright_ti5"]]
                .mean()
                .reset_index()
                .rename(columns={"acq_date": "index"})
                .set_index("index")
            )
            st.subheader("Average brightness")
            st.line_chart(data=viz_df)

            viz_df = (
                data.groupby("acq_date")["confidence"]
                .value_counts()
                .unstack(fill_value=0)
                .reset_index()
                .rename(columns={"acq_date": "index"})
                .set_index("index")
            )
            st.subheader("Confidence levels")
            st.bar_chart(data=viz_df)
        else:
            st.warning("No data found for the query!", icon="⚠️")


def main():
    llm = ChatOpenAI(temperature=0)
    parser = DialogParser(llm=llm)

    run_streamlit(parser)
    # debug()


if __name__ == "__main__":
    main()
