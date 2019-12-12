# encoding: utf8
from __future__ import unicode_literals

from ..symbols import *
from ..language_data import PRON_LEMMA


MORPH_RULES = {
    "PRP": {
        "I":            {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Sing", "Case": "Nom"},
        "me":           {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Sing", "Case": "Acc"},
        "you":          {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Two"},
        "he":           {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Masc", "Case": "Nom"},
        "him":          {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Masc", "Case": "Acc"},
        "she":          {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Fem",  "Case": "Nom"},
        "her":          {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Fem",  "Case": "Acc"},
        "it":           {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Neut"},
        "we":           {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Plur", "Case": "Nom"},
        "us":           {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Plur", "Case": "Acc"},
        "they":         {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Plur", "Case": "Nom"},
        "them":         {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Plur", "Case": "Acc"},

        "mine":         {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Sing", "Poss": "Yes", "Reflex": "Yes"},
        "yours":        {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Two", "Poss": "Yes", "Reflex": "Yes"},
        "his":          {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Masc", "Poss": "Yes", "Reflex": "Yes"},
        "hers":         {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Fem",  "Poss": "Yes", "Reflex": "Yes"},
        "its":          {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Gender": "Neut", "Poss": "Yes", "Reflex": "Yes"},
        "ours":         {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Plur", "Poss": "Yes", "Reflex": "Yes"},
        "yours":        {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Two", "Number": "Plur", "Poss": "Yes", "Reflex": "Yes"},
        "theirs":       {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Plur", "Poss": "Yes", "Reflex": "Yes"},

        "myself":       {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Sing",  "Case": "Acc", "Reflex": "Yes"},
        "yourself":     {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Two", "Case": "Acc", "Reflex": "Yes"},
        "himself":      {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Case": "Acc", "Gender": "Masc", "Reflex": "Yes"},
        "herself":      {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Case": "Acc", "Gender": "Fem",  "Reflex": "Yes"},
        "itself":       {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Case": "Acc", "Gender": "Neut", "Reflex": "Yes"},
        "themself":     {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Sing", "Case": "Acc", "Reflex": "Yes"},
        "ourselves":    {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "One", "Number": "Plur", "Case": "Acc", "Reflex": "Yes"},
        "yourselves":   {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Two", "Case": "Acc", "Reflex": "Yes"},
        "themselves":   {LEMMA: PRON_LEMMA, "PronType": "Prs", "Person": "Three", "Number": "Plur", "Case": "Acc", "Reflex": "Yes"}
    },

    "PRP$": {
        "my":           {LEMMA: PRON_LEMMA, "Person": "One", "Number": "Sing", "PronType": "Prs", "Poss": "Yes"},
        "your":         {LEMMA: PRON_LEMMA, "Person": "Two", "PronType": "Prs", "Poss": "Yes"},
        "his":          {LEMMA: PRON_LEMMA, "Person": "Three", "Number": "Sing", "Gender": "Masc", "PronType": "Prs", "Poss": "Yes"},
        "her":          {LEMMA: PRON_LEMMA, "Person": "Three", "Number": "Sing", "Gender": "Fem",  "PronType": "Prs", "Poss": "Yes"},
        "its":          {LEMMA: PRON_LEMMA, "Person": "Three", "Number": "Sing", "Gender": "Neut", "PronType": "Prs", "Poss": "Yes"},
        "our":          {LEMMA: PRON_LEMMA, "Person": "One", "Number": "Plur", "PronType": "Prs", "Poss": "Yes"},
        "their":        {LEMMA: PRON_LEMMA, "Person": "Three", "Number": "Plur", "PronType": "Prs", "Poss": "Yes"}
    },

    "VBZ": {
        "am":           {LEMMA: "be", "VerbForm": "Fin", "Person": "One", "Tense": "Pres", "Mood": "Ind"},
        "are":          {LEMMA: "be", "VerbForm": "Fin", "Person": "Two", "Tense": "Pres", "Mood": "Ind"},
        "is":           {LEMMA: "be", "VerbForm": "Fin", "Person": "Three", "Tense": "Pres", "Mood": "Ind"},
    },

    "VBP": {
        "are":          {LEMMA: "be", "VerbForm": "Fin", "Tense": "Pres", "Mood": "Ind"}
    },

    "VBD": {
        "was":          {LEMMA: "be", "VerbForm": "Fin", "Tense": "Past", "Number": "Sing"},
        "were":         {LEMMA: "be", "VerbForm": "Fin", "Tense": "Past", "Number": "Plur"}
    }
}
