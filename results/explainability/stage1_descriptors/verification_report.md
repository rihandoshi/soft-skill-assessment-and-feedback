# Stage 1 verification

Automated evidence for the Stage 1 completion checklist (`scripts/explainability/verify_stage1.py`). Human listening is still required; this narrows down what to listen for. Duplicate recordings excluded.

## Checklist

| # | Item | Automated result | Remaining for a human |
|---|---|---|---|
| 1 | Listen to the 5 most extreme examples in each of the 7 categories | Listening sheet annotated with Whisper text/rate, pYIN agreement and audibility verdicts (`listening_check/listening_check_annotated.csv`) | **Listen and fill in `heard_as_extreme`** |
| 2 | Fastest/slowest sentences genuinely representative | Fastest: 21/25 still above train p95 (225 WPM) when re-timed by Whisper; slowest: 20/25 still below p5 (95 WPM). See §A | Confirm the disagreeing cases by ear |
| 3 | Expressive examples free of pitch-tracking artefacts | Praat vs pYIN octave disagreement in the most expressive answers: median 9.8%, max 31.9% (controls: median 0.4%). See §B | Listen to the top 5 expressive |
| 4 | Final rises and trailing off audible | Trailing off: {'audible: voice drops, tail still speech': 25}. Final rise: {'audible rise (>= 3 st)': 25}. See §C | Listen to the top 5 of each |
| 5 | -300 dB artefacts cannot enter descriptors | Injection test: max change 0.08 dB (mean), 0.06 dB (std), 0.89 dB (end drop), vs -18.0 dB if they were not excluded. See §D | None |
| 6 | Review extraction report and missing values | §E | Read `extraction_report.md` |

## A. Speech-rate extremes re-timed with Whisper

Each sentence's VAD-trimmed span is re-counted with faster-whisper word timestamps, independent of the RecruitView transcript and its 1-s timestamps. `word_ratio` = Whisper words / transcript words in the span.

- Control sentences (random 40): Spearman(transcript WPM, Whisper WPM) = 0.77; median word ratio 0.94.
- Fastest 25: median transcript 300 WPM vs Whisper 255 WPM; median word ratio 0.86; 21 remain extreme.
- Slowest 25: median transcript 60 WPM vs Whisper 77 WPM; median word ratio 1.25; 20 remain extreme.

| group | id | start-end (s) | transcript WPM | Whisper WPM | word ratio | transcript text | Whisper text |
|---|---|---|---|---|---|---|---|
| fastest | 1492 | 10.0-15.0 | 336 | 372 | 1.11 | And I put it in my brain that I have to go to the gym, I have to put o | very skinny kid. And I put it in my brain that I have to go to the gym |
| fastest | 1612 | 39.5-44.0 | 333 | 307 | 0.92 | I feel that that is one of the greatest failures of my life till date  | I feel that that is one of the greatest failures of my life till date, |
| fastest | 0912 | 43.0-47.0 | 330 | 15 | 0.05 | i like to code stuff that is my i think that is what my dream company  | to |
| fastest | 0936 | 38.0-42.0 | 330 | 315 | 0.95 | That is what I like and that is I think I think that is why I think I  | before deadline. That is what I like and that is I think that is why I |
| fastest | 1758 | 39.0-45.0 | 320 | 270 | 0.84 | so like if one is saying that I am going to collect data set then the  | the same time so like if one is saying that I'm going to collect data  |
| fastest | 1732 | 11.0-15.0 | 315 | 240 | 0.76 | so I don't really panic like a lot of people do when things don't go a | schedule so I don't really panic like a lot of people do when things d |
| fastest | 1562 | 17.1-22.0 | 306 | 220 | 0.72 | Half the team was not divided on the part but they were trying to get  | How the team was not divided on the part, but they were trying to get  |
| fastest | 0402 | 5.0-9.0 | 300 | 255 | 0.85 | I have to wake up at this time and I have to complete my syllabus by t | like I have to wake up at this time and I have to complete my syllabus |
| fastest | 0414 | 13.0-17.0 | 300 | 240 | 0.80 | machine learning. So I had to come out of my comfort zone and in the m | learning machine learning so I had to come out of my comfort zone and  |
| fastest | 0822 | 27.0-33.0 | 300 | 310 | 1.03 | and then listing up on the pros and cons of the situation, that what w | it thoroughly and then listing up on the pros and cons of the situatio |
| slowest | 1291 | 32.0-44.0 | 50 | 50 | 1.00 | So having good performance in my studies makes me happy. | So having good performance in my studies makes me happy. |
| slowest | 0061 | 23.0-36.9 | 52 | 52 | 1.00 | I am currently studying to do the GRE and the IELTS exam. | I am currently studying to do the GRE and the IS exam |
| slowest | 1671 | 34.0-40.8 | 53 | 79 | 1.50 | in any unconditioned or unpredicted situation. | people around us in any unconditioned or unpredicted situation. |
| slowest | 0649 | 39.0-46.9 | 53 | 106 | 2.00 | My weakness would be that I procrastinate. | me and my weakness would be that my weaknesses would be that I procras |
| slowest | 0108 | 54.0-59.5 | 55 | 65 | 1.20 | blockchain and all. Thank you. | security blockchain and all thank you |
| slowest | 0367 | 22.0-35.0 | 55 | 69 | 1.25 | gaming and working out and I am extremely passionate about coding and | my hobbies include gaming and working out and I am extremely passionat |
| slowest | 1083 | 10.0-34.6 | 56 | 105 | 1.87 | After selecting the topic, we have to get feedback from the team membe | team member and even after selecting the topic, for further how the pr |
| slowest | 0153 | 20.0-31.7 | 56 | 72 | 1.27 | And I also love to read books, novels, especially the thrillers. | Jim Frick also. And I also love to read books, novels, especially the  |
| slowest | 0417 | 11.0-16.0 | 60 | 120 | 2.00 | basic JavaScript and HTML CSS. | had done front -end, basics of JavaScripts and HTML CSS. |
| slowest | 0656 | 24.0-30.0 | 60 | 70 | 1.17 | I also have a positive attitude. | they arise. I also have a positive |

Full table: `verification/rate_extremes_whisper.csv`.

## B. Pitch: Praat vs pYIN

`octave_disagreement_frac` = share of frames voiced in both trackers whose F0 differs by > 6 st (an octave error in one of them).

| Measure | Value |
|---|---|
| spearman_praat_vs_pyin_std | 0.651 |
| control: median octave disagreement | 0.004 |
| control: max octave disagreement | 0.101 |
| control: median Praat std | 1.810 |
| control: median pYIN std | 2.361 |
| most expressive: median octave disagreement | 0.098 |
| most expressive: max octave disagreement | 0.319 |
| most expressive: median Praat std | 3.735 |
| most expressive: median pYIN std | 6.057 |
| most monotone: median octave disagreement | 0.002 |
| most monotone: max octave disagreement | 0.058 |
| most monotone: median Praat std | 1.131 |
| most monotone: median pYIN std | 2.191 |

| group | id | speaker | Praat std (st) | pYIN std (st) | octave disagreement | median abs diff (st) |
|---|---|---|---|---|---|---|
| most expressive | 1603 | 42 | 4.29 | 7.44 | 13.9% | 0.19 |
| most expressive | 1866 | 7 | 4.23 | 10.60 | 31.9% | 0.25 |
| most expressive | 1846 | 2 | 4.04 | 4.47 | 4.1% | 0.19 |
| most expressive | 0114 | 7 | 3.99 | 7.62 | 11.1% | 0.17 |
| most expressive | 1591 | 42 | 3.76 | 7.62 | 10.6% | 0.20 |
| most expressive | 1795 | 7 | 3.71 | 9.03 | 9.1% | 0.19 |
| most expressive | 0425 | 35 | 3.69 | 4.68 | 12.6% | 0.18 |
| most expressive | 1785 | 2 | 3.67 | 4.10 | 3.0% | 0.14 |
| most expressive | 0297 | 128 | 3.66 | 3.86 | 0.7% | 0.17 |
| most expressive | 1879 | 2 | 3.58 | 3.79 | 1.3% | 0.17 |
| most monotone | 0010 | 101 | 1.06 | 2.73 | 3.8% | 0.12 |
| most monotone | 0023 | 164 | 1.36 | 1.43 | 0.0% | 0.13 |
| most monotone | 0049 | 94 | 1.07 | 1.27 | 0.0% | 0.07 |
| most monotone | 0068 | 72 | 1.32 | 1.51 | 1.1% | 0.09 |
| most monotone | 0110 | 254 | 1.21 | 1.75 | 0.0% | 0.14 |
| most monotone | 0149 | 19 | 1.02 | 2.59 | 5.8% | 0.09 |
| most monotone | 0170 | 236 | 1.25 | 3.96 | 2.2% | 0.09 |
| most monotone | 0205 | 40 | 1.14 | 2.42 | 0.1% | 0.09 |
| most monotone | 0215 | 111 | 1.13 | 1.96 | 0.2% | 0.08 |
| most monotone | 0255 | 248 | 0.95 | 3.55 | 0.2% | 0.09 |

## C. Audibility of trailing off and final rises

`tail_above_noise_db` = syllable-peak level in the last 0.7 s minus the clip's background level (p10 of non-speech frames). < 10 dB means the 'speech' at the end is really background. Trailing off counts as audible at a drop of >= 3 dB with the tail still speech; a final rise at >= 3 st over 300 ms.

| group | id | span (s) | end drop (dB) | rise (st/s) | tail above noise (dB) | verdict | text |
|---|---|---|---|---|---|---|---|
| strongest trailing off | 1779 | 0.8-8.3 | -28.3 | nan | 22.9 | audible: voice drops, tail still speech | I learn new skills either from YouTube or platform that offe |
| strongest trailing off | 0681 | 15.0-24.0 | -23.1 | -3.3 | 15.3 | audible: voice drops, tail still speech | that same time and while also working on myself and my techn |
| strongest trailing off | 1019 | 42.0-45.6 | -22.2 | -3.9 | 42.1 | audible: voice drops, tail still speech | I constantly look for ways to improve my learning processes. |
| strongest trailing off | 1676 | 51.0-53.7 | -20.9 | 3.0 | 25.5 | audible: voice drops, tail still speech | So I think that's an example of my creativity. |
| strongest trailing off | 0044 | 26.0-29.0 | -20.5 | -2.8 | 24.6 | audible: voice drops, tail still speech | opportunities and challenges that lie ahead in front of my l |
| strongest trailing off | 0954 | 44.1-50.9 | -20.2 | -5.0 | 94.9 | audible: voice drops, tail still speech | Yeah, but we managed it. So I think we can... I personally c |
| strongest trailing off | 0490 | 39.5-44.3 | -20.0 | -5.4 | 19.1 | audible: voice drops, tail still speech | Pertaining to that they've lied to me and done something wro |
| strongest trailing off | 1798 | 14.0-18.5 | -20.0 | -1.7 | 15.1 | audible: voice drops, tail still speech | comes on at once and you have to pull an all-nighter to fini |
| strongest final rise | 1496 | 40.0-43.6 | -3.7 | 69.9 | 29.0 | audible rise (>= 3 st) | That shows my adaptability and leadership skills. |
| strongest final rise | 1457 | 36.0-40.0 | -6.3 | 67.9 | 31.5 | audible rise (>= 3 st) | where everyone did work that was based on their own strength |
| strongest final rise | 1190 | 0.6-6.0 | -3.9 | 67.0 | 35.4 | audible rise (>= 3 st) | Yes, I can provide an excellent example of when I demonstrat |
| strongest final rise | 0167 | 6.0-11.0 | -15.1 | 64.6 | 48.9 | audible rise (>= 3 st) | BTech and Computer Science and Engineering from Manipal Univ |
| strongest final rise | 2004 | 16.0-19.0 | -0.6 | 62.7 | 31.9 | audible rise (>= 3 st) | That's a huge motivating factor. |
| strongest final rise | 1496 | 28.0-32.0 | -2.7 | 61.0 | 32.4 | audible rise (>= 3 st) | I am proud of my software development skills. |
| strongest final rise | 0772 | 4.0-8.0 | -1.1 | 57.6 | 40.4 | audible rise (>= 3 st) | So I will answer that with a yes. |
| strongest final rise | 0406 | 24.2-31.0 | -7.6 | 57.2 | 20.4 | audible rise (>= 3 st) | But still I had worked and got to know about it more at that |

Full table: `verification/audibility.csv`.

## D. Digital-silence dropouts (-300 dB)

- Clips with any dropout inside VAD speech: 44; with > 1% of speech frames: 9; max 2.7%.
- Code path: every intensity statistic uses `int_ok = speech & ~dropout`; the intensity sampled at pitch frames (terminal-slope guard) sets dropout frames to NaN; QC plots now mask them and mark them with a rug.
- Injection test (20 clean clips, 5% of speech frames set to -300 dB, real `describe_clip` re-run): max change 0.08 dB in mean intensity, 0.06 dB in intensity std, 0.89 dB in end drop. A naive mean over the same frames would shift by -18.0 dB.
- Observed ranges of every dB descriptor:

| descriptor | min | max |
|---|---|---|
| `sentence.intensity_mean_db` | 33.2 | 80.9 |
| `sentence.intensity_std_db` | 2.4 | 30.9 |
| `sentence.end_drop_db` | -28.3 | 21.0 |
| `segment.intensity_mean_db` | 33.2 | 78.8 |
| `segment.intensity_std_db` | 3.5 | 22.9 |
| `segment.end_drop_median_db` | -28.3 | 8.1 |

## E. Missing values (train, unique recordings, answer level)

| descriptor | status | NaN % | reason when missing |
|---|---|---|---|
| `rushed_opening_ratio` | active | 38.9 | needs >= 3 sentences, first two and the rest each >= 4 s |
| `sent_rate_std_wpm` | active | 25.9 | needs >= 2 sentences that are >= 4 s and >= 5 words |
| `sent_rate_cv` | active | 25.9 | needs >= 2 sentences that are >= 4 s and >= 5 words |
| `rate_drift_ratio` | active | 18.6 | needs a speech span >= 12 s |
| `monotony_frac` | experimental | 5.2 | needs a speech span >= 5 s with >= 1 s voiced per window |
| `sent_rate_max_wpm` | active | 5.2 | needs >= 1 sentence >= 4 s and >= 5 words |
| `sent_rate_mean_wpm` | active | 5.2 | needs >= 1 sentence >= 4 s and >= 5 words |
| `artic_rate_wpm` | active | 4.6 | no transcript / no VAD speech / too little voicing |
| `speech_rate_wpm` | active | 4.6 | no transcript / no VAD speech / too little voicing |
| `artic_rate_sps` | active | 4.6 | no transcript / no VAD speech / too little voicing |
| `terminal_slope_median_st_s` | experimental | 4.1 | needs >= 10 strong voiced frames at a sentence end |
| `end_drop_median_db` | experimental | 2.7 | no transcript / no VAD speech / too little voicing |
| `pitch_std_st` | experimental | 2.0 | no transcript / no VAD speech / too little voicing |
| `f0_median_hz` | diagnostic | 2.0 | no transcript / no VAD speech / too little voicing |
| `f0_median_st` | experimental | 2.0 | no transcript / no VAD speech / too little voicing |
| `pitch_range_st` | experimental | 2.0 | no transcript / no VAD speech / too little voicing |
| `intensity_mean_db` | diagnostic | 1.8 | no transcript / no VAD speech / too little voicing |
| `intensity_std_db` | experimental | 1.8 | no transcript / no VAD speech / too little voicing |
| `long_mid_pause_rate_per_min` | active | 1.4 | no transcript / no VAD speech / too little voicing |
| `mid_pause_max_s` | active | 1.4 | no transcript / no VAD speech / too little voicing |
| `between_pause_time_ratio` | active | 1.4 | no transcript / no VAD speech / too little voicing |
| `mid_pause_time_ratio` | active | 1.4 | no transcript / no VAD speech / too little voicing |
| `mid_pause_rate_per_min` | active | 1.4 | no transcript / no VAD speech / too little voicing |
| `voiced_frac` | diagnostic | 1.3 | no transcript / no VAD speech / too little voicing |
| `pause_ratio` | active | 1.3 | no transcript / no VAD speech / too little voicing |
| `speech_span_s` | active | 1.3 | no transcript / no VAD speech / too little voicing |
| `dropout_frac` | diagnostic | 1.3 | no transcript / no VAD speech / too little voicing |
| `response_latency_s` | active | 1.3 | no transcript / no VAD speech / too little voicing |
| `octave_jump_frac` | diagnostic | 1.2 | no transcript / no VAD speech / too little voicing |
| `jitter_local` | experimental | 0.4 | no transcript / no VAD speech / too little voicing |
| `shimmer_local` | experimental | 0.4 | no transcript / no VAD speech / too little voicing |
| `audio_duration_s` | active | 0.0 | no transcript / no VAD speech / too little voicing |
| `syllable_count` | diagnostic | 0.0 | no transcript / no VAD speech / too little voicing |
| `n_sentences` | active | 0.0 | no transcript / no VAD speech / too little voicing |
| `n_rate_sentences` | diagnostic | 0.0 | no transcript / no VAD speech / too little voicing |
| `word_count` | active | 0.0 | no transcript / no VAD speech / too little voicing |
| `speech_time_s` | active | 0.0 | no transcript / no VAD speech / too little voicing |
| `internal_pause_count` | diagnostic | 0.0 | no transcript / no VAD speech / too little voicing |
| `n_monotony_windows` | diagnostic | 0.0 | no transcript / no VAD speech / too little voicing |
| `isolated_blips_removed` | diagnostic | 0.0 | no transcript / no VAD speech / too little voicing |
| `pitch_floor_hz` | diagnostic | 0.0 | no transcript / no VAD speech / too little voicing |
| `pitch_ceiling_hz` | diagnostic | 0.0 | no transcript / no VAD speech / too little voicing |
