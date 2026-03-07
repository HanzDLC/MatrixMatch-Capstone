from app import app
import json
with app.app_context():
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user_id'] = 23
            sess['first_name'] = 'Hanz'
            sess['last_name'] = 'de la Cruz'
            sess['role'] = 'Admin'
        res = client.get('/api/online_users')
        print("STATUS:", res.status_code)
        if res.status_code == 200:
            print("DATA:", res.json)
        else:
            print("ERROR HTML:", res.data[:500])
