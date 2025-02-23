# Hackathon Template

Instructions:
1. This repository must be used as a template to create your teams' submission repository.
2. The Check-In section must remain intact, and you must edit it to include your details. Please indicate agreement to the terms by makring the checklist with an `[x]`.
3. The final demo video can be included as a file in the submission repository, or as a publicly accessible video on any website (e.g., Youtube).
4. All other sections, including this, can be edited as you see fit - including removing these instructions for your submission.

# Check-In

- Title of your submission: **[Fine-tuning pretrained scGPT for bioinformatics service in the biotech industry]**
- Team Members: [Hong Qin](mailto:hqin@odu.edu), [AMAN AL AMIN](mailto:malam007@odu.edu), [Terry Stilwell](mailto:tstilwel@odu.edu), [Min Dong](mailto:mdong@odu.edu), [Yaping Feng](mailto:yaping.feng@admerahealth.com)
- [X] All team members agree to abide by the [Hackathon Rules](https://aaai.org/conference/aaai/aaai-25/hackathon/)
- [X] This AAAI 2025 hackathon entry was created by the team during the period of the hackathon, February 17 – February 24, 2025
- [X] The entry includes a 2-minute maximum length demo video here: [Link](https://youtu.be/RC0v8jZNhFk)
- [X] The entry clearly identifies the selected theme in the README and the video.

# Seletected Theme: Advanced Algorithms in Practice
The project of Team 75, Fine-tuning pretrained scGPT for bioinformatics service in the biotech industry, is a collaboration between Old Dominion University and Admera Health LLC (https://www.admerahealth.com/) 

We aim to finetune scGPT (https://github.com/bowang-lab/scGPT) to cell type identifing using the single cell RNAseq data. 
The long-term goal is to integrate this capability into the bioinformatics pipeline of Admera Health. 

Our project aligns with theme "Advanced Algorithms in Practice: Bring cutting-edge AI methodologies like reinforcement learning, active learning, or foundation models into real-world applications."

# Hackathon AAAI homepage
https://aaai.org/conference/aaai/aaai-25/hackathon/?utm_source=chatgpt.com

# data are stored in largedata directory, need to add manually. 

# Run scGPT on ODU Wahab
To use scGPT, please follow these commands:

module load pytorch-gpu/2.1 

crun.pytorch-gpu -p ~/envs/scGPT python test_scgpt_wahab.py




