# Model State File
# Bu dosya model aktivasyonlarında güncellenir ve Flask debug mode tarafından izlenir
# Otomatik restart için config.py tarafından import edilir

MODEL_STATE = {
    'age': {
        'active_version': 4,
        'last_activation': '2025-05-27T18:09:14.403369'
    },
    'content': {
        'active_version': None,
        'last_activation': None
    }
}

# Bu satır Flask'ın dosya değişikliklerini algılaması için
# Her model aktivasyonunda timestamp güncellenir
LAST_UPDATE = "2025-05-27T18:09:14.403369"