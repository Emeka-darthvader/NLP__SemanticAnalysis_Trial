from grakn.client import GraknClient

with GraknClient(uri="localhost:48555") as client:
    with client.session(keyspace="grakn") as session:
        ## Insert a Person using a WRITE transaction
        # with session.transaction().write() as write_transaction:
        #     insert_iterator = write_transaction.query('insert $Franky isa Person, has Gender "Male",has accountNumber "000233444";')
        #     concepts = insert_iterator.collect_concepts()
        #     print("Inserted a person with ID: {0}".format(concepts[0].id))
        #     ## to persist changes, write transaction must always be committed (closed)
        #     write_transaction.commit()

        # Read the person using a READ only transaction
        # with session.transaction().read() as read_transaction:
        #     answer_iterator = read_transaction.query("match $x isa Person; get; limit 10;")
        #     # print(answer)
        #     for answer in answer_iterator:
        #         # print(answer)
        #         person= answer.map().get("x")
        #         print(person.id)
        #         # person = answer.map().get("Male")
        #         # print("Retrieved person with id " + person.id)
        #         # print("Retrieved person with id " + person)

        ## Or query and consume the iterator immediately collecting all the results
        with session.transaction().read() as read_transaction:
            answer_iterator = read_transaction.query("match $x isa Person; get; limit 10;")
            #concept_map = next(answer_iterator)
            
            # Get Entity Obj from ConceptMap
            #entity = concept_map.map()['x']
            #print("Entity Obj: {}".format(entity))

            #print("Entity Id: {}".format(entity.id))

            # Get Entity Attribute Values
            #attrs = entity.attributes()
            #for each in attrs:
            #    print(each.value())
            persons = answer_iterator.collect_concepts()
            for person in persons:
                print("Retrieved person with id "+ person.id)
            print(vars(answer_iterator))

            

        ## if not using a `with` statement, then we must always close the session and the read transaction
        # read_transaction.close()
        # session.close()
        # client.close()