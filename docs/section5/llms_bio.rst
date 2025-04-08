Large Language Models in Biology
================================

Here we go over hands-on examples of how Large Language Models (LLMs) are being
used in life sciences computing applications. The examples are driven by current
topics found in the literature.


Example 1: Protein Design
-------------------------

Goal: Design a novel protein with a specific function

Approach: ProtGPT2 is an LLM trained on protein sequences.

Hands on:

First install HuggingFace transformer python package

.. code-block:: console

    $ pip install transformers 


Then generate de novo proteins: 

.. code-block:: python3

    >>> from transformers import pipeline
    >>> protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")
    # length is expressed in tokens, where each token has an average length of 4 amino acids.
    >>> sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)
    >>> for seq in sequences:
            print(seq):
    {'generated_text': 'MINDLLDISRIISGKMTLDRAEVNLTAIARQVVEEQRQAAEAKSIQLLCSTPDTNHYVFG\nDFDRLKQTLWNLLSNAVKFTPSGGTVELELGYNAEGMEVYVKDSGIGIDPAFLPYVFDRF\nRQSDAADSRNYGGLGLGLAIVKHLLDLHEGNVSAQSEGFGKGATFTVLLPLKPLKRELAA\nVNRHTAVQQSAPLNDNLAGMKILIVEDRPDTNEMVSYILEEAGAIVETAESGAAALTSLK\nSYSPDLVLSDIGMPMMDGYEMIEYIREWKTTKGG'}
    {'generated_text': 'MQGDSSISSSNRMFT\nLCKPLTVANETSTLSTTRNSKSNKRVSKQRVNLAESPERNAPSPASIKTNETEEFSTIKT\nTNNEVLGYEPNYVSYDFVPMEKCNLCNENCSIELASLNEETFVKKTICCHECRKKAIENA\nENNNTKGSAVSNNSVTSSSGRKKIIVSGSQILRNLDSLTSSKSNISTLLNPNHLAKLAKN\nGNLSSLSSLQSSASSISKSSSTSSTPTTSPKVSSPTNSPSSSPINSPTP'}
    {'generated_text': 'M\nSTHVSLENTLASLQATFFSLEARHTALETQLLSTRTELAATKQELVRVQAEISRADAQAQ\nDLKAQILTLKEKADQAEVEAAAATQRAEESQAALEAQTAELAQLRLEKQAPQHVAEEGDP\nQPAAPTTQAQSPVTSAAAAASSAASAEPSKPELTFPAYTKRKPPTITHAPKAPTKVALNP\nSTLSTSGSGGGAKADPTPTTPVPSSSAGLIPKALRLPPPVTPAASGAKPAPSARSKLRGP\nDAPLSPSTQS'}
    {'generated_text': 'MVLLSTGPLPILFLGPSLAELNQKYQVVSDTLLRFTNTV\nTFNTLKFLGSDS\n'}
    {'generated_text': 'M\nNNDEQPFIMSTSGYAGNTTSSMNSTSDFNTNNKSNTWSNRFSNFIAYFSGVGWFIGAISV\nIFFIIYVIVFLSRKTKPSGQKQYSRTERNNRDVDSIKRANYYG\n'}
    {'generated_text': 'M\nEAVYSFTITETGTGTVEVTPLDRTISGADIVYPPDTACVPLTVQPVINANGTWTLGSGCT\nGHFSVDTTGHVNCLTGGFGAAGVHTVIYTVETPYSGNSFAVIDVNVTEPSGPGDGGNGNG\nDRGDGPDNGGGNNPGPDPDPSTPPPPGDCSSPLPVVCSDRDCADFDTQAQVQIYLDRYGG\nTCDLDGNHDGTPCENLPNNSGGQSSDSGNGGGNPGTGSTHQVVTGDCLWNIASRNNGQGG\nQAWPALLAANNESITNP'}
    {'generated_text': 'M\nGLTTSGGARGFCSLAVLQELVPRPELLFVIDRAFHSGKHAVDMQVVDQEGLGDGVATLLY\nAHQGLYTCLLQAEARLLGREWAAVPALEPNFMESPLIALPRQLLEGLEQNILSAYGSEWS\nQDVAEPQGDTPAALLATALGLHEPQQVAQRRRQLFEAAEAALQAIRASA\n'}
    {'generated_text': 'M\nGAAGYTGSLILAALKQNPDIAVYALNRNDEKLKDVCGQYSNLKGQVCDLSNESQVEALLS\nGPRKTVVNLVGPYSFYGSRVLNACIEANCHYIDLTGEVYWIPQMIKQYHHKAVQSGARIV\nPAVGFDSTPAELGSFFAYQQCREKLKKAHLKIKAYTGQSGGASGGTILTMIQHGIENGKI\nLREIRSMANPREPQSDFKHYKEKTFQDGSASFWGVPFVMKGINTPVVQRSASLLKKLYQP\nFDYKQCFSFSTLLNSLFSYIFNAI'}
    {'generated_text': 'M\nKFPSLLLDSYLLVFFIFCSLGLYFSPKEFLSKSYTLLTFFGSLLFIVLVAFPYQSAISAS\nKYYYFPFPIQFFDIGLAENKSNFVTSTTILIFCFILFKRQKYISLLLLTVVLIPIISKGN\nYLFIILILNLAVYFFLFKKLYKKGFCISLFLVFSCIFIFIVSKIMYSSGIEGIYKELIFT\nGDNDGRFLIIKSFLEYWKDNLFFGLGPSSVNLFSGAVSGSFHNTYFFIFFQSGILGAFIF\nLLPFVYFFISFFKDNSSFMKLF'}
    {'generated_text': 'M\nRRAVGNADLGMEAARYEPSGAYQASEGDGAHGKPHSLPFVALERWQQLGPEERTLAEAVR\nAVLASGQYLLGEAVRRFETAVAAWLGVPFALGVASGTAALTLALRAYGVGPGDEVIVPAI\nTFIATSNAITAAGARPVLVDIDPSTWNMSVASLAARLTPKTKAILAVHLWGQPVDMHPLL\nDIAAQANLAVIEDCAQALGASIAGTKVGTFGDAAAFSFYPTKNMTTGEGGMLVTNARDLA\nQAARMLRSHGQDPPTAYMHSQVGFN'}



Cite: 

`ProtGPT2 is a deep unsupervised language model for protein design <https://www.nature.com/articles/s41467-022-32007-7>`_

Data and model:

`on Hugging Face <https://huggingface.co/nferruz/ProtGPT2>`_


Example 2: Drug Discovery
-------------------------

Goal: Design new small molecule drugs for treating disease

Approach: Use Generative Chemical Language models (CLMs) for de novo design

Hands on:

First set up the environment 

.. codeblock:: console

    $ git clone https://github.com/ETHmodlab/hybridCLMs
    $ cd hybridCLMs
    $ conda env create -f environment.yml


Then run the code to generate new molecules:

.. code-block:: python3

    >>> from hybridCLM import CLM
    >>> clm = CLM()
    >>> clm.load_model('chembl')
    >>> clm.generate_molecules(num_molecules=10)
    ['CC(=O)C1=CC=C(C=C1)C(=O)O', 'CC(=O)C1=CC=C(C=C1)C(=O)N', ...]

    >>> clm.evaluate_molecules(molecules=['CC(=O)C1=CC=C(C=C1)C(=O)O', 'CC(=O)C1=CC=C(C=C1)C(=O)N'])
    [{'molecule': 'CC(=O)C1=CC=C(C=C1)C(=O)O', 'score': 0.85}, {'molecule': 'CC(=O)C1=CC=C(C=C1)C(=O)N', 'score': 0.78}, ...]



Cite:

`Leveraging molecular structure and bioactivity with chemical language models for de novo drug design <https://www.nature.com/articles/s41467-022-35692-6>`_

Data and model: 

`on zenodo <https://zenodo.org/records/7370858>`_

`on GitHub <https://github.com/ETHmodlab/hybridCLMs/tree/v1.0>`_



Example 3: DNA Sequence Context
-------------------------------

Goal: Predict the next nucleotides in a DNA sequence given the context

Approach: Use DNA Language Model GROVER trained on human DNA sequences

First install HuggingFace transformer python package

.. code-block:: console

    $ pip install transformers 


Then generate de novo DNA sequences:

Maybe follow `this tutorial <https://zenodo.org/records/8373159>`_

Cite: 

`DNA language model GROVER learns sequence context in the human genome <https://www.nature.com/articles/s42256-024-00872-0>`_

Data and model:

`on Hugging Face <https://huggingface.co/PoetschLab/GROVER>`_

`source code <https://zenodo.org/records/13374192>`_



References
----------

This is a good summary paper on the use of LLMs in biology:

`Large language models and their applications in bioinformatics <https://www.sciencedirect.com/science/article/pii/S2001037024003209>`_